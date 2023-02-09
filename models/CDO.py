import torch
from models.resnet import *
import numpy as np
import os
from models.hrnet.hrnet import HRNet_
from torch.nn import functional as F
import loguru
from scipy.ndimage import gaussian_filter

valid_resnet_backbones = ['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2']
valid_hrnet_backbones = ['hrnet18', 'hrnet32', 'hrnet48']

class CDOModel(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CDOModel, self).__init__()

        self.out_size_h = kwargs['out_size_h']
        self.out_size_w = kwargs['out_size_w']

        self.device = kwargs['device']
        self.gamma = kwargs['gamma']
        self.OOM = kwargs['OOM']

        self.model = self.get_model(**kwargs)

    def get_model(self, **kwargs) ->torch.nn.Module:
        backbone = kwargs['backbone']

        if backbone in valid_resnet_backbones:
            # expert network
            model_expert, _ = eval(f'{backbone}(pretrained=True)')
            # apprentice network
            model_apprentice, _ = eval(f'{backbone}(pretrained=False)')
        elif backbone in valid_hrnet_backbones:
            # expert network
            model_expert = HRNet_(backbone, pretrained=True)
            # apprentice network
            model_apprentice = HRNet_(backbone, pretrained=False)
        else:
            raise NotImplementedError

        # fix the parameters of the expert network
        for param in model_expert.parameters():
            param.requires_grad = False

        model_expert.eval()
        model = torch.nn.ModuleDict({'ME': model_expert, 'MA': model_apprentice})

        return model

    def forward(self, x)->dict:

        features = dict()
        # map the input to the expert domain and extract corresponding features
        with torch.no_grad():
            features['FE'] = self.model['ME'](x)

        # map the input to the apprentice domain and extract corresponding features
        features['FA'] = self.model['MA'](x)

        return features

    def save(self, path, metric):
        torch.save(self.model['MA'].state_dict(), path)

    def load(self, path):
        self.model['MA'].load_state_dict(torch.load(path, map_location=self.device))

    def train_mode(self):
        self.model['ME'].eval()
        self.model['MA'].train()

    def eval_mode(self):
        self.model['ME'].eval()
        self.model['MA'].eval()

    def cal_discrepancy(self, fe, fa, OOM, normal, gamma, aggregation=True):
        # normalize the features into uint vector
        fe = F.normalize(fe, p=2, dim=1)
        fa = F.normalize(fa, p=2, dim=1)

        # calculate feature-to-feature discrepancy d_p
        d_p = torch.sum((fe - fa) ** 2, dim=1)

        if OOM:
            # if OOM is utilized, we need to calculate the adaptive weights for individual features

            # calculate the mean discrepancy \mu_p to indicate the importance of individual features
            mu_p = torch.mean(d_p)

            if normal:
                # for normal samples: w = ((d_p) / \mu_p)^{\gamma}
                w = (d_p / mu_p) ** gamma

            else:
                # for abnormal samples: w = ((d_p) / \mu_p)^{-\gamma}
                w = (mu_p / d_p) ** gamma

            w = w.detach()

        else:
            # else, we manually assign each feature the same weight, i.e., 1
            w = torch.ones_like(d_p)

        if aggregation:
            d_p = torch.sum(d_p * w)

        sum_w = torch.sum(w)

        return d_p, sum_w

    def cal_loss(self, fe_list, fa_list, gamma=2, mask=None):
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

            loss_n, weight_n = self.cal_discrepancy(fe_n, fa_n, OOM=self.OOM, normal=True, gamma=gamma,
                                                   aggregation=True)
            loss_s, weight_s = self.cal_discrepancy(fe_s, fa_s, OOM=self.OOM, normal=False, gamma=gamma,
                                                   aggregation=True)

            # L= {{\textstyle \sum_{i=1}^{N_n}}{{(w_n)_i}d(p_n)_i}-{\textstyle \sum_{j=1}^{N_a}}{{(w_s)_j}d(p_s)_j}} /
            # {{\textstyle \sum_{i=1}^{N_n}}{{(w_n)_i}}+{\textstyle \sum_{j=1}^{N_a}}{{(w_s)_j}}}
            loss += ((loss_n - loss_s) / (weight_n + weight_s) * B)

        return loss

    @torch.no_grad()
    def cal_am(self, **kwargs):
        fe_list = kwargs['FE']
        fa_list = kwargs['FA']

        anomaly_map = 0

        # interpolate the feature map into the size of the first hierarchy
        H_0, W_0 = fe_list[0].shape[2], fe_list[0].shape[3]
        for i in range(len(fe_list)):
            fe_list[i] = F.interpolate(fe_list[i], size=(H_0, W_0), mode='bilinear', align_corners=True)
            fa_list[i] = F.interpolate(fa_list[i], size=(H_0, W_0), mode='bilinear', align_corners=True)

        for fe, fa in zip(fe_list, fa_list):
            _, _, h, w = fe.shape

            a_map, _ = self.cal_discrepancy(fe, fa, gamma=self.gamma, aggregation=False, OOM=False,
                                           normal=False)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            # fuse anomaly maps from different hierarchies
            anomaly_map += a_map

        am_np = anomaly_map.squeeze(1).cpu().numpy()

        am_np_list = []

        for i in range(am_np.shape[0]):
            am_np[i] = gaussian_filter(am_np[i], sigma=4)
            am_np_list.append(am_np[i])

        return am_np_list





