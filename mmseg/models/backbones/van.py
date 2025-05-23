# code from: https://github.com/Visual-Attention-Network/VAN-Segmentation/blob/main/van.py

import torch
import torch.nn as nn

from timm.models.layers import DropPath
# from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
#                                         trunc_normal_init)
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from torch.nn.modules.utils import _pair as to_2tuple
# from mmseg.models.builder import BACKBONES
from mmseg.registry import MODELS as BACKBONES
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint

from mmcv.cnn import build_norm_layer
# from mmcv.runner import BaseModule
from mmengine.model import BaseModule
import math
import warnings


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


@BACKBONES.register_module()
class VAN(BaseModule):
    def __init__(self,
                 in_chans=3,
                #  embed_dims=[64, 128, 256, 512],
                 embed_dims=[32, 64, 160, 256],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_rate=0.,
                #  drop_path_rate=0.,
                 drop_path_rate=0.2,
                #  depths=[3, 4, 6, 3],
                 depths=[3, 3, 5, 2],
                 num_stages=4,
                 linear=False,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 **kwargs):
        super(VAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.pretrained = init_cfg['checkpoint']
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(dim=embed_dims[i],
                                         mlp_ratio=mlp_ratios[i],
                                         drop=drop_rate,
                                         drop_path=dpr[cur + j],
                                         linear=linear,
                                         norm_cfg=norm_cfg)
                                   for j in range(depths[i])])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self, pretrained=None):
        if isinstance(self.pretrained, str) and self.pretrained != "":
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, self.pretrained, map_location='cpu', strict=False, logger=logger)

    # def init_weights(self):
    #     print('init cfg', self.init_cfg)
    #     if self.init_cfg is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 trunc_normal_init(m, std=.02, bias=0.)
    #             elif isinstance(m, nn.LayerNorm):
    #                 constant_init(m, val=1.0, bias=0.)
    #             elif isinstance(m, nn.Conv2d):
    #                 fan_out = m.kernel_size[0] * m.kernel_size[
    #                     1] * m.out_channels
    #                 fan_out //= m.groups
    #                 # normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
    #     else:
    #         super(VAN, self).init_weights()

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x