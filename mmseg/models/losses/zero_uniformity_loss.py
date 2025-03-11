import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
from mmseg.registry import MODELS
from .utils import weighted_loss
from torch import Tensor


@weighted_loss
def zero_uniformity_loss(pred: Tensor, target: Tensor):
    if (pred.shape[0] > 1): return torch.zeros([pred.shape[0]])
    if len(pred.shape) == 4: # intra
        mean = pred.mean(dim=[2, 3])
        std = mean.std(dim=[1])
    loss = std
    return loss




__factory__ = {
    "zero_uniformity_loss": zero_uniformity_loss,
}


@MODELS.register_module()
class ZeroUniformityLoss(nn.Module):
    def __init__(self, 
                 loss_type="zero_uniformity_loss",
                 reduction='mean', 
                 loss_weight=1.0, 
                 ignore_index=-1, 
                 loss_name="loss_ZUL"):
        super(ZeroUniformityLoss, self).__init__()
        self.isFirstWarning = True
        if ignore_index != -1 and self.isFirstWarning:
            self.isFirstWarning = False
            warnings.warn(f"warning: {loss_name} hasn't implemented ignore_index, value: {ignore_index}")
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.loss_type = loss_type

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-1):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        if ignore_index != -1 and self.isFirstWarning:
            self.isFirstWarning = False
            warnings.warn(f"warning: {self.loss_name} hasn't implemented ignore_index, value: {ignore_index}")
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * __factory__[self.loss_type](
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss

    @property
    def loss_name(self):
        return self._loss_name
