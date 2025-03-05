import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from . import LOSS


@LOSS.register_module
class CDOLoss(nn.Module):
    def __init__(self, gamma=2, OOM=True):
        super(CDOLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        self.gamma = gamma
        self.OOM = OOM

    def cal_discrepancy(self, fe, fa, normal, aggregation=True):
        # normalize the features into uint vector
        fe = F.normalize(fe, p=2, dim=1)
        fa = F.normalize(fa, p=2, dim=1)

        # calculate feature-to-feature discrepancy d_p
        d_p = torch.sum((fe - fa) ** 2, dim=1)

        if self.OOM:
            # if OOM is utilized, we need to calculate the adaptive weights for individual features

            # calculate the mean discrepancy \mu_p to indicate the importance of individual features
            mu_p = torch.mean(d_p)

            if normal:
                # for normal samples: w = ((d_p) / \mu_p)^{\gamma}
                w = (d_p / mu_p) ** self.gamma

            else:
                # for abnormal samples: w = ((d_p) / \mu_p)^{-\gamma}
                w = (mu_p / d_p) ** self.gamma

            w = w.detach()

        else:
            # else, we manually assign each feature the same weight, i.e., 1
            w = torch.ones_like(d_p)

        if aggregation:
            d_p = torch.sum(d_p * w)

        sum_w = torch.sum(w)

        return d_p, sum_w

    def forward(self, fe_list, fa_list, mask=None):
        loss = 0

        # interpolate the feature map into the size of the first hierarchy
        B, _, H_0, W_0 = fe_list[0].shape
        for i in range(len(fe_list)):
            fe_list[i] = F.interpolate(fe_list[i], size=(H_0, W_0), mode='bilinear', align_corners=True)
            fa_list[i] = F.interpolate(fa_list[i], size=(H_0, W_0), mode='bilinear', align_corners=True)

        for fe, fa in zip(fe_list, fa_list):

            B, C, H, W = fe.shape

            # if mask is inputted, then we collaboratively optimize the discrepancies for normal and abnormal samples
            if mask is not None:
                mask_vec = F.interpolate(mask, (H, W), mode='nearest')
            else:
                mask_vec = torch.zeros((B, C, H, W))

            # reshape the mask, fe, and fa the the same shape for easily index
            mask_vec = mask_vec.permute(0, 2, 3, 1).reshape(-1, )

            fe = fe.permute(0, 2, 3, 1).reshape(-1, C)
            fa = fa.permute(0, 2, 3, 1).reshape(-1, C)

            # process normal and abnormal samples individually

            # normal features
            fe_n = fe[mask_vec == 0]
            fa_n = fa[mask_vec == 0]

            # synthetic abnormal features
            fe_s = fe[mask_vec != 0]
            fa_s = fa[mask_vec != 0]

            loss_n, weight_n = self.cal_discrepancy(fe_n, fa_n, normal=True,
                                                   aggregation=True)
            loss_s, weight_s = self.cal_discrepancy(fe_s, fa_s, normal=False,
                                                   aggregation=True)

            # L= {{\textstyle \sum_{i=1}^{N_n}}{{(w_n)_i}d(p_n)_i}-{\textstyle \sum_{j=1}^{N_a}}{{(w_s)_j}d(p_s)_j}} /
            # {{\textstyle \sum_{i=1}^{N_n}}{{(w_n)_i}}+{\textstyle \sum_{j=1}^{N_a}}{{(w_s)_j}}}
            loss += ((loss_n - loss_s) / (weight_n + weight_s) * B)

        return loss
