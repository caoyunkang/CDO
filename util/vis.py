import copy
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import font_manager

matplotlib.use('Agg')
import seaborn as sns
import os


def vis_rgb_gt_amp(img_paths, imgs, img_masks, anomaly_maps_, method, root_out):
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)

    anomaly_maps = copy.deepcopy(anomaly_maps_)
    for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = "_".join(needed_parts)
        # specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}'
        os.makedirs(out_dir, exist_ok=True)
        img_path = f'{out_dir}/{specific_root}_{img_num}_img.png'
        img_ano_path = f'{out_dir}/{specific_root}_{img_num}_amp.png'
        mask_path = f'{out_dir}/{specific_root}_{img_num}_mask.png'

        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
        img_rec = img * std[:, None, None] + mean[:, None, None]
        # RGB image
        img_rec = Image.fromarray((img_rec * 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0))
        img_rec.save(img_path)
        # RGB image with anomaly map

        if method == 'adaclip':
            cur_max_value = np.percentile(anomaly_map, 95)
            cur_min_value = np.percentile(anomaly_map, 5)

            scale_factor = 255 / (cur_max_value - cur_min_value)
            anomaly_map = ((anomaly_map - cur_min_value) * scale_factor).round().clip(0, 255).astype(np.uint8)
        else:
            anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-9)

        anomaly_map = cm.jet(anomaly_map)
        # anomaly_map = cm.rainbow(anomaly_map)
        anomaly_map = (anomaly_map[:, :, :3] * 255).astype('uint8')
        anomaly_map = Image.fromarray(anomaly_map)
        img_rec_anomaly_map = Image.blend(img_rec, anomaly_map, alpha=0.5)
        img_rec_anomaly_map.save(img_ano_path)
        # mask
        img_mask = Image.fromarray((img_mask * 255).astype(np.uint8).transpose(1, 2, 0).repeat(3, axis=2))
        img_rec_mask = Image.blend(img_rec, img_mask, alpha=0.5)
        img_rec_mask.save(mask_path)


def plot_two_pair_distri(a_before=None, n_before=None, a_after=None, n_after=None,
                         save_root='experiments/score_distributions',
                         save_name='temp.png', category='皮革', normalize=True):
    plt.figure(figsize=(4, 3))
    # plt.figure(figsize=(12, 4))
    font_size = 24
    config = {
        # "font.family": 'serif',
        "font.size": font_size,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun'],
    }
    plt.rcParams.update(config)

    simsun_font = font_manager.FontProperties(fname="./fonts/simsun.ttc")
    new_roman__font = font_manager.FontProperties(fname="./fonts/Times New Roman.ttf")

    data_all = []
    if n_before:
        data_all.append(np.load(n_before))
    if a_before:
        data_all.append(np.load(a_before))
    if n_after:
        data_all.append(np.load(n_after))
    if a_after:
        data_all.append(np.load(a_after))

    data_all = np.concatenate(data_all)
    max_value = np.max(data_all)
    min_value = np.min(data_all)

    if a_before:
        a_before = np.load(a_before)

        if normalize:
            a_before = (a_before - min_value) / (max_value - min_value + 1e-7)
        # sns.kdeplot(a_before,color="red",label='KDE of abnormal points $\mathbf{w/o}$ CPS', linestyle='-')
        sns.kdeplot(a_before, color="red", linestyle='-')

    if n_before:
        n_before = np.load(n_before)
        if normalize:
            n_before = (n_before - min_value) / (max_value - min_value + 1e-7)
        # sns.kdeplot(n_before,color="green",label='KDE of normal points $\mathbf{w/o}$ CPS',linestyle='-')
        sns.kdeplot(n_before, color="green", linestyle='-')

    if a_after:
        a_after = np.load(a_after)
        if normalize:
            a_after = (a_after - min_value) / (max_value - min_value + 1e-7)
        # sns.kdeplot(a_after,color="orange",label='KDE of abnormal points $\mathbf{w/}$ CPS', linestyle='--')
        sns.kdeplot(a_after, color="orange", linestyle='--')

    if n_after:
        n_after = np.load(n_after)
        if normalize:
            n_after = (n_after - min_value) / (max_value - min_value + 1e-7)
        # sns.kdeplot(n_after,color="blue",label='KDE of normal points $\mathbf{w/}$ CPS', linestyle='--')
        sns.kdeplot(n_after, color="blue", linestyle='--')

    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.title(category, fontsize=font_size, fontproperties=simsun_font)
    # plt.xlabel('Anomaly Score', fontsize=16)
    # plt.ylabel('Density', fontsize=16)
    plt.xlabel('异常分值', fontsize=font_size, fontproperties=simsun_font)
    plt.ylabel('密度', fontsize=font_size, fontproperties=simsun_font)
    plt.xticks(fontsize=font_size, fontproperties=new_roman__font)
    plt.yticks(fontsize=font_size, fontproperties=new_roman__font)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 5.05))
    # plt.legend(loc="upper right")
    # plt.legend(None)

    os.makedirs(save_root, exist_ok=True)
    savepath = os.path.join(save_root, save_name)
    plt.savefig(savepath, dpi=600, bbox_inches='tight',pad_inches=0.1)
    # plt.savefig(savepath, dpi=600,)
    plt.close()
    print(f"Save fig to {savepath}")
