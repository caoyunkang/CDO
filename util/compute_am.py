from scipy.ndimage import gaussian_filter
import numpy as np
import torch
from torch.nn import functional as F


def compute_discrepancy_map(ft_list, fs_list, out_size=[224, 224], uni_am=False,
                            use_cos=True, amap_mode='add', gaussian_sigma=0, weights=None):
    bs = ft_list[0].shape[0]
    weights = weights if weights else [1] * len(ft_list)
    anomaly_map = np.ones([bs] + out_size) if amap_mode == 'mul' else np.zeros([bs] + out_size)
    a_map_list = []
    if uni_am:
        size = (ft_list[0].shape[2], ft_list[0].shape[3])
        for i in range(len(ft_list)):
            ft_list[i] = F.interpolate(F.normalize(ft_list[i], p=2), size=size, mode='bilinear', align_corners=True)
            fs_list[i] = F.interpolate(F.normalize(fs_list[i], p=2), size=size, mode='bilinear', align_corners=True)
        ft_map, fs_map = torch.cat(ft_list, dim=1), torch.cat(fs_list, dim=1)
        if use_cos:
            a_map = 1 - F.cosine_similarity(ft_map, fs_map, dim=1)
            a_map = a_map.unsqueeze(dim=1)
        else:
            a_map = torch.sqrt(torch.sum((ft_map - fs_map) ** 2, dim=1, keepdim=True))
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map.squeeze(dim=1).cpu().detach().numpy()
        anomaly_map = a_map
        a_map_list.append(a_map)
    else:
        for i in range(len(ft_list)):
            ft = ft_list[i]
            fs = fs_list[i]
            if use_cos:
                a_map = 1 - F.cosine_similarity(ft, fs, dim=1)
                a_map = a_map.unsqueeze(dim=1)
            else:
                a_map = torch.sqrt(torch.sum((ft - fs) ** 2, dim=1, keepdim=True))
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            a_map = a_map.squeeze(dim=1)
            a_map = a_map.cpu().detach().numpy()
            a_map_list.append(a_map)
            if amap_mode == 'add':
                anomaly_map += a_map * weights[i]
            else:
                anomaly_map *= a_map
        if amap_mode == 'add':
            anomaly_map /= (len(ft_list) * sum(weights))

    if gaussian_sigma > 0:
        for idx in range(anomaly_map.shape[0]):
            anomaly_map[idx] = gaussian_filter(anomaly_map[idx], sigma=gaussian_sigma)
    return anomaly_map, a_map_list


def maximum_as_anomaly_score(anomaly_map, max_ratio=0.):
    anomaly_map_flatten = anomaly_map.reshape(anomaly_map.shape[0], -1)

    if max_ratio == 0:
        sp_score = np.max(anomaly_map_flatten, axis=1)
    else:
        sorted_scores = np.sort(anomaly_map_flatten, axis=1)[:, ::-1]
        sp_score = sorted_scores[:, :int(anomaly_map_flatten.shape[1] * max_ratio)]
        sp_score = np.mean(sp_score, axis=1)

    return sp_score
