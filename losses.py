import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

class SRN_loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(SRN_loss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.first = True
        self.resize = partial(F.interpolate, mode='area', recompute_scale_factor=True)

    def forward(self, batch_p, batch_l):
        assert batch_p[0].shape[0] == batch_l.shape[0]
        device = batch_p[0].device
        b, c, h, w = batch_p[0].shape

        loss = self.loss_weight * self.scale * torch.log(
            ((batch_p[0] - batch_l) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss += 0.5 * self.loss_weight * self.scale * torch.log(
            ((batch_p[1] - self.resize(input=batch_l, scale_factor=0.5)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss += 0.25 * self.loss_weight * self.scale * torch.log(
            ((batch_p[2] - self.resize(input=batch_l, scale_factor=0.25)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        loss += 0.125 * self.loss_weight * self.scale * torch.log(
            ((batch_p[3] - self.resize(input=batch_l, scale_factor=0.125)) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

        # mlwnet returned loss, 0.0 but let's just return loss for simpler usage
        return loss
