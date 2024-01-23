import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import NECKS
from mmcv.cnn.weight_init import caffe2_xavier_init
from ..utils import window_partition, MixerBlock,  ConvModule


@NECKS.register_module
class MLPCTXFPN(nn.Module):
    """MLPCTXFPN

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_dim=8,
                 feat_channels=[4, 16, 128, 1024],
                 mixer_count=1,
                 ):
        super(MLPCTXFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.mixer_count = mixer_count
        self.patch_dim = patch_dim
        self.feat_channels = feat_channels

        self.mapper = nn.ModuleList()
        self.ctx = nn.ModuleList()
        for i in range(len(self.feat_channels)):
            self.mapper.append(ConvModule(self.in_channels[i], self.feat_channels[i], 1))
            self.ctx.append(ConvModule(self.feat_channels[i], 8, 3, dilation=i+1, padding=i+1))

        self.mixers = None
        if self.mixer_count > 0:
            self.mixers = nn.Sequential(*[
                MixerBlock(self.patch_dim**2, self.out_channels) for i in range(self.mixer_count)
            ])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):

        B, H4, W4, _ = inputs[0].shape
        parts = []

        for i in range(len(self.feat_channels)):
            part = self.mapper[i](inputs[i])
            if i > 0:
                part = F.interpolate(part, scale_factor=2**i, mode='bilinear')
            part = self.ctx[i](part)
            parts.append(part)

        out = torch.cat(parts, dim=1)
        out = window_partition(out, self.patch_dim, channel_last=False)

        B, T = out.shape[:2]
        outputs = out.view(B, T, self.patch_dim**2, self.out_channels)

        if self.mixers is not None:
            outputs = self.mixers(outputs)

        return tuple([outputs])
