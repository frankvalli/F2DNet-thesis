import torch.nn as nn

from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule
import torch.nn.functional as F
import torch
from mmdet.core import (delta2bbox, multiclass_nms, bbox_target, force_fp32, bbox2result,
                        auto_fp16)


@HEADS.register_module
class RefineHead(BBoxHead):
    """More general shared bbox head, with dropouts.
    """

    def __init__(self,
                 num_shared_convs=2,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=2,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 use_gm=True,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 alpha=0.5,
                 num_classes=1,
                 weight_decay=0.0005,
                 dropout_prob=-1,
                 *args,
                 **kwargs):
        super(RefineHead, self).__init__(*args, num_classes=num_classes, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.alpha = alpha
        self.use_gm = use_gm
        self.dropout_prob = dropout_prob
        self.reg_lambda = weight_decay

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= (self.roi_feat_size * self.roi_feat_size)
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= (self.roi_feat_size * self.roi_feat_size)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
                if self.dropout_prob > -1:
                    branch_fcs.append(nn.Dropout(self.dropout_prob))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(RefineHead, self).init_weights()
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def compute_reg_loss(self):
        reg_loss = torch.tensor(0.0).float().cuda()
        for param in self.parameters():
            reg_loss += param.square().sum()
        return dict(loss_reg=self.reg_lambda * reg_loss)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        if self.num_classes <= 1:
            cls_score = cls_score.view(-1)
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def get_scores(self, x):

        cls_score, _ = self.forward(x)

        if cls_score.dim() > 1:
            return F.softmax(cls_score, dim=1)
        return torch.sigmoid(cls_score)

    def suppress_boxes(self, rois, scores, img_meta=None, cfg=None):
        img_shape = img_meta[0]['img_shape']

        bboxes = rois[:, 1:].clone()
        if img_shape is not None:
            bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
            bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return [bbox2result(det_bboxes, det_labels, self.num_classes)[0]]

    def combine_scores(self, results, scores):
        results = results[0]
        if scores.dim() > 1:
            scores = scores[:, -1]
        if not self.use_gm:
            results[:, 4] *= self.alpha
            results[:, 4] += (1.0 - self.alpha) * scores
        else:
            results[:, 4] *= scores
        return [results.cpu().numpy()]
