from mmcv.runner import Runner

import logging
import os.path as osp
import time

import mmcv
import torch

from mmcv.runner import hooks
from mmcv.runner.log_buffer import LogBuffer
from mmdet.core.my_mmcv.runner.hooks.mean_teacher_optimizer import OptimizerHook
from mmcv.runner.hooks import (Hook, LrUpdaterHook, CheckpointHook, IterTimerHook, lr_updater)
from mmcv.runner.checkpoint import load_checkpoint, save_checkpoint
from mmcv.runner.priority import get_priority
from mmcv.runner.utils import get_dist_info, get_host_info, get_time_str, obj_from_dict
from collections import OrderedDict


class Mean_teacher_Runner(Runner):
    """mean teacher runner.

        Args:
            model (:obj:`torch.nn.Module`): The model to be run.
            batch_processor (callable): A callable method that process a data
                batch. The interface of this method should be
                `batch_processor(model, data, train_mode) -> dict`
            optimizer (dict or :obj:`torch.optim.Optimizer`): If it is a dict,
                runner will construct an optimizer according to it.
            work_dir (str, optional): The working directory to save checkpoints
                and logs.
            log_level (int): Logging level.
            logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
                default logger.
        """

    def __init__(self,
                 model,
                 batch_processor,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None,
                 mean_teacher=False):
        assert callable(batch_processor)
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.batch_processor = batch_processor
        self.teacher_dict = {}
        self.student_dict = {}
        self.mean_teacher = mean_teacher

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._iters_epoch = None
        self._max_iters = 0

    def load_teacher_dict_to_model(self):
        print("Loading teacher dict for evaluation ...")
        self.student_dict = self.weights_to_cpu(self.model.module.state_dict())
        self.model.module.load_state_dict(self.teacher_dict)

    def load_student_dict_to_model(self):
        print("Loading back student dict for training ...")
        self.model.module.load_state_dict(self.student_dict)
        self.model = self.model.cuda()

    def load_mean_teacher_checkpoint(self, cfg):
        if cfg.load_from or cfg.resume_from:
            if cfg.load_from:
                checkpoint = torch.load(cfg.load_from + '.stu')
                self.teacher_dict = checkpoint['state_dict']
                for k in self.model.module.state_dict():
                    if not k in self.teacher_dict:
                        self.teacher_dict[k] = self.model.module.state_dict()[k]
            if cfg.resume_from:
                checkpoint = torch.load(cfg.resume_from + '.stu')
                self.teacher_dict = checkpoint['state_dict']
            for k, v in self.teacher_dict.items():
                self.teacher_dict[k] = self.teacher_dict[k].cuda()
            return

        self.teacher_dict = dict()
        for k, v in self.model.module.state_dict().items():
            self.teacher_dict[k] = v

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, 'latest.pth')
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        if self.mean_teacher:
            mean_teacher_path = filepath + ".stu"
            self.save_mean_teacher_checkpoint(self.teacher_dict, mean_teacher_path)
        # use relative symlink
        mmcv.symlink(filename, linkpath)

    def save_mean_teacher_checkpoint(self, state_dict,filename):
        checkpoint = {
            'state_dict': self.weights_to_cpu(state_dict)
        }
        torch.save(checkpoint, filename)

    def weights_to_cpu(self, state_dict):
        """Copy a model state_dict to cpu.

        Args:
            state_dict (OrderedDict): Model weights on GPU.

        Returns:
            OrderedDict: Model weights on GPU.
        """
        state_dict_cpu = OrderedDict()
        for key, val in state_dict.items():
            state_dict_cpu[key] = val.cpu()
        return state_dict_cpu

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())

        if log_config is not None:
            self.register_logger_hooks(log_config)

    def current_lr(self):
        """Get current learning rates.
        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        if isinstance(self.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.optimizer.param_groups]
        elif isinstance(self.optimizer, dict):
            lr = dict()
            for name, optim in self.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def current_momentum(self):
        """Get current momentums.
        Returns:
            list[float] | dict[str, list[float]]: Current momentums of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """

        def _get_momentum(optimizer):
            momentums = []
            for group in optimizer.param_groups:
                if 'momentum' in group.keys():
                    momentums.append(group['momentum'])
                elif 'betas' in group.keys():
                    momentums.append(group['betas'][0])
                else:
                    momentums.append(0)
            return momentums

        if self.optimizer is None:
            raise RuntimeError(
                'momentum is not applicable because optimizer does not exist.')
        elif isinstance(self.optimizer, torch.optim.Optimizer):
            momentums = _get_momentum(self.optimizer)
        elif isinstance(self.optimizer, dict):
            momentums = dict()
            for name, optim in self.optimizer.items():
                momentums[name] = _get_momentum(optim)
        return momentums

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        if self._iters_epoch is not None:
            self._max_iters = self._max_epochs * self._iters_epoch
        else:
            self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1
            if self._iters_epoch != None and self._inner_iter >= self._iters_epoch:
                break

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs, iters_epoch=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        self._iters_epoch = iters_epoch 
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epochs:
                        return
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')