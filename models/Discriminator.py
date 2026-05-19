import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class AdversarialNetwork(nn.Module):
    def __init__(self, in_channels_list):
        super(AdversarialNetwork, self).__init__()
        self.num_classes = 2
        self.adv_heads = nn.ModuleList()
        self.in_channels_list = [c * self.num_classes for c in in_channels_list]

        for in_features in self.in_channels_list:
            self.adv_heads.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(in_features, 128, kernel_size=4, stride=2, padding=1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1))
                )
            )

    def forward(self, x, coeff, layer_index):
        if coeff > 0:
            x.register_hook(grl_hook(coeff))
        if layer_index < len(self.adv_heads):
            return self.adv_heads[layer_index](x)
        else:
            raise IndexError("AdversarialNetwork: layer_index out of bounds.")


class BoundedChangeContrastiveLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.05):
        super(BoundedChangeContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6

    def forward(self, feat1, feat2, change_mask):
        feat1_norm = F.normalize(feat1, p=2, dim=1)
        feat2_norm = F.normalize(feat2, p=2, dim=1)
        feat1_flat = feat1_norm.view(feat1_norm.size(0), feat1_norm.size(1), -1)
        feat2_flat = feat2_norm.view(feat2_norm.size(0), feat2_norm.size(1), -1)
        change_mask_resized = F.interpolate(change_mask.float(), size=feat1_norm.shape[2:], mode='nearest')
        mask_flat = change_mask_resized.view(change_mask_resized.size(0), -1)
        dist_sq = torch.sum(torch.pow(feat1_flat - feat2_flat, 2), dim=1)
        dist = torch.sqrt(dist_sq + self.eps)

        loss_changed = F.relu(self.alpha - dist) * mask_flat
        loss_unchanged = F.relu(dist - self.beta) * (1 - mask_flat)

        num_pixels = dist.numel()
        return torch.sum(loss_changed + loss_unchanged) / num_pixels if num_pixels > 0 else 0.0