import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from . import LOSS


def compute_lcs_in_chunks(Qe, Qa, chunk_size):
    N, _ = Qe.shape
    l_cs_sum = 0.0
    count = 0

    for i in range(0, N, chunk_size):
        Qe_chunk = Qe[i:i + chunk_size]
        Qa_chunk = Qa[i:i + chunk_size]

        # 对角块
        Ge_chunk = Qe_chunk @ Qe_chunk.T
        Ga_chunk = Qa_chunk @ Qa_chunk.T
        l_cs_sum += ((Ge_chunk - Ga_chunk) ** 2).sum()
        count += Ge_chunk.numel()

        # 非对角块
        for j in range(i + chunk_size, N, chunk_size):
            Qe_other = Qe[j:j + chunk_size]
            Qa_other = Qa[j:j + chunk_size]

            Ge_off = Qe_chunk @ Qe_other.T
            Ga_off = Qa_chunk @ Qa_other.T
            l_cs_sum += ((Ge_off - Ga_off) ** 2).sum()
            count += Ge_off.numel()

    return l_cs_sum / count



@LOSS.register_module
class IKDLoss(nn.Module):
    def __init__(self, beta=2, gamma=1, alpha=0.01):
        super(IKDLoss, self).__init__()

        self.margin = None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.chunk_size = 1024
        self.binary_map = []

    def cal_ikd_hierarchy(self, fe, fa, indx):
        BS, C, H, W = fe.shape

        # normalize the features into uint vector
        fe = F.normalize(fe, p=2, dim=1)
        fa = F.normalize(fa, p=2, dim=1)

        ####### AHSM
        dif = (fe - fa) ** 2
        dif = torch.sum(dif, dim=1)

        # calculate the margin for hard samples mining
        mu = dif.mean().item()
        variance = dif.std().item()

        # exponential moving average
        self.margin[indx] = self.alpha * self.margin[indx] + (1 - self.alpha) * (mu + self.beta * variance)

        # select_idx = dif >= self.margin[indx]
        # self.binary_map.append(select_idx.float())

        select_idx = dif >= self.margin[indx]
        select_idx = select_idx.view(-1)
        if select_idx.sum() == 0:
            print(f'No elements are selected in IKD loss, return average')
            return torch.mean(dif)

        ####### Pixel-wise similarity loss
        dif = dif[dif >= self.margin[indx]]
        l_ps = torch.mean(dif)


        if self.gamma > 0:
            ###### Context similarity loss
            Qe = fe.permute(0, 2, 3, 1).reshape(-1, C)
            Qa = fa.permute(0, 2, 3, 1).reshape(-1, C)

            Qe = Qe[select_idx, :]
            Qa = Qa[select_idx, :]

            # print(Qe.numel())
            Ge = Qe @ Qe.T
            Ga = Qa @ Qa.T

            l_cs = (Ge - Ga) ** 2
            l_cs = l_cs.mean()

            # l_cs = compute_lcs_in_chunks(Qe, Qa, chunk_size=self.chunk_size)

            loss = l_ps + self.gamma * l_cs
        else:
            loss = l_ps

        return loss

    def forward(self, fe_list, fa_list):
        self.binary_map = []
        if self.margin is None:
            self.margin = [0. for _ in fe_list]

        loss = 0

        for indx, (fe, fa) in enumerate(zip(fe_list, fa_list)):
            loss += self.cal_ikd_hierarchy(fe, fa, indx)

        return loss
