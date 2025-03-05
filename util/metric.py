from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure
import multiprocessing
import copy

import numpy as np
from numba import jit
import torch
from torch.nn import functional as F

from util.util import get_timepc, log_msg
from util.registry import Registry

from adeval import EvalAccumulatorCuda

EVALUATOR = Registry('Evaluator')


def func(th, amaps, binary_amaps, masks):
    print('start', th)
    binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
    pro = []
    for binary_amap, mask in zip(binary_amaps, masks):
        for region in measure.regionprops(measure.label(mask)):
            tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
            pro.append(tp_pixels / region.area)
    inverse_masks = 1 - masks
    fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
    fpr = fp_pixels / inverse_masks.sum()
    print('end', th)
    return [np.array(pro).mean(), fpr, th]


class Evaluator(object):
    def __init__(self, metrics=[], pooling_ks=None, max_step_aupro=200, mp=False, use_adeval=False):
        ## TODO: check the metrics we required.
        if len(metrics) == 0:
            self.metrics = [
                'mAUROC_sp_max', 'mAUROC_px', 'mAUPRO_px',
                'mAP_sp_max', 'mAP_px',
                'mF1_max_sp_max',
                'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
                'mF1_max_px', 'mIoU_max_px',
            ]
        else:
            self.metrics = metrics

        self.pooling_ks = pooling_ks
        self.max_step_aupro = max_step_aupro
        self.mp = mp

        self.eps = 1e-8
        self.beta = 1.0

        self.boundary = 1e-9
        self.use_adeval = use_adeval

    def run(self, results, cls_name, logger=None):
        idxes = results['cls_names'] == cls_name
        gt_px = results['imgs_masks'][idxes].squeeze(1)
        pr_px = results['anomaly_maps'][idxes]
        gt_sp = results['anomalys'][idxes]
        pr_sp = results['anomaly_scores'][idxes]

        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)

        # normalization for pixel-level evaluations
        pr_px_norm = (pr_px - pr_px.min()) / (pr_px.max() - pr_px.min())
        gt_sp = gt_px.max(axis=(1, 2))

        if self.use_adeval:
            score_min = min(pr_sp) - self.boundary
            score_max = max(pr_sp) + self.boundary
            anomap_min = pr_px.min() - self.boundary
            anomap_max = pr_px.max() + self.boundary
            accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=50)
            accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                                   torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))
            for i in range(torch.tensor(pr_px).size(0)):
                accum.add_image(torch.tensor(pr_sp[i]), torch.tensor(gt_sp[i]))
            metrics = accum.summary()

        metric_str = f'==> Metric Time for {cls_name:<15}: '
        metric_results = {}
        for metric in self.metrics:
            t0 = get_timepc()
            if metric.startswith('mAUROC_sp_max'):
                auroc_sp = roc_auc_score(gt_sp, pr_sp)
                metric_results[metric] = auroc_sp
            elif metric.startswith('AUROC_sp'):
                auroc_sp = roc_auc_score(gt_sp, pr_sp)
                metric_results[metric] = auroc_sp
            elif metric.startswith('mAUROC_px'):
                if not self.use_adeval:
                    auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
                    metric_results[metric] = auroc_px
                else:
                    metric_results[metric] = metrics['p_auroc']
            elif metric.startswith('mAUPRO_px'):
                if not self.use_adeval:
                    aupro_px = self.cal_pro_score(gt_px, pr_px, max_step=self.max_step_aupro, mp=self.mp)
                    metric_results[metric] = aupro_px
                else:
                    metric_results[metric] = metrics['p_aupro']
            elif metric.startswith('mAP_sp_max'):
                ap_sp = average_precision_score(gt_sp, pr_sp)
                metric_results[metric] = ap_sp
            elif metric.startswith('mAP_px'):
                if not self.use_adeval:
                    ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
                    metric_results[metric] = ap_px
                else:
                    metric_results[metric] = metrics['p_aupr']
            elif metric.startswith('mF1_max_sp_max'):
                precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
                f1_scores = (2 * precisions * recalls) / (precisions + recalls)
                best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_threshold = thresholds[best_f1_score_index]
                metric_results[metric] = best_f1_score
            elif metric.startswith('mF1_px') or metric.startswith('mDice_px') or metric.startswith('mAcc_px') or metric.startswith('mIoU_px'):  # example: F1_sp_0.3_0.8
                # F1_px equals Dice_px
                coms = metric.split('_')
                assert len(coms) == 5, f"{metric} should contain parameters 'score_l', 'score_h', and 'score_step' "
                score_l, score_h, score_step = float(metric.split('_')[-3]), float(metric.split('_')[-2]), float(metric.split('_')[-1])
                gt = gt_px.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_px_norm > score
                    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
                    total_area_union = np.logical_or(gt, pr).sum(axis=(0, 1, 2))
                    total_area_pred_label = pr.sum(axis=(0, 1, 2))
                    total_area_label = gt.sum(axis=(0, 1, 2))
                    if metric.startswith('mF1_px'):
                        precision = total_area_intersect / (total_area_pred_label + self.eps)
                        recall = total_area_intersect / (total_area_label + self.eps)
                        f1_px = (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall + self.eps)
                        metric_scores.append(f1_px)
                    elif metric.startswith('mDice_px'):
                        dice_px = 2 * total_area_intersect / (total_area_pred_label + total_area_label + self.eps)
                        metric_scores.append(dice_px)
                    elif metric.startswith('mAcc_px'):
                        acc_px = total_area_intersect / (total_area_label + self.eps)
                        metric_scores.append(acc_px)
                    elif metric.startswith('mIoU_px'):
                        iou_px = total_area_intersect / (total_area_union + self.eps)
                        metric_scores.append(iou_px)
                    else:
                        raise f'invalid metric: {metric}'
                metric_results[metric] = np.array(metric_scores).mean()
            elif metric.startswith('mF1_max_px') or metric.startswith('mDice_max_px') or metric.startswith('mAcc_max_px') or metric.startswith('mIoU_max_px'):
                # F1_px equals Dice_px
                score_l, score_h, score_step = 0.0, 1.0, 0.05
                gt = gt_px.astype(np.bool_)
                metric_scores = []
                for score in np.arange(score_l, score_h + 1e-3, score_step):
                    pr = pr_px_norm > score
                    total_area_intersect = np.logical_and(gt, pr).sum(axis=(0, 1, 2))
                    total_area_union = np.logical_or(gt, pr).sum(axis=(0, 1, 2))
                    total_area_pred_label = pr.sum(axis=(0, 1, 2))
                    total_area_label = gt.sum(axis=(0, 1, 2))
                    if metric.startswith('mF1_max_px'):
                        precision = total_area_intersect / (total_area_pred_label + self.eps)
                        recall = total_area_intersect / (total_area_label + self.eps)
                        f1_px = (1 + self.beta ** 2) * precision * recall / (
                                    self.beta ** 2 * precision + recall + self.eps)
                        metric_scores.append(f1_px)
                    elif metric.startswith('mDice_max_px'):
                        dice_px = 2 * total_area_intersect / (total_area_pred_label + total_area_label + self.eps)
                        metric_scores.append(dice_px)
                    elif metric.startswith('mAcc_max_px'):
                        acc_px = total_area_intersect / (total_area_label + self.eps)
                        metric_scores.append(acc_px)
                    elif metric.startswith('mIoU_max_px'):
                        iou_px = total_area_intersect / (total_area_union + self.eps)
                        metric_scores.append(iou_px)
                    else:
                        raise f'invalid metric: {metric}'
                metric_results[metric] = np.array(metric_scores).max()
            t1 = get_timepc()
            metric_str += f'{t1 - t0:7.3f} ({metric})\t'
        log_msg(logger, metric_str)
        return metric_results

    @staticmethod
    def cal_pro_thr(results, th, amaps, masks):
        binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        results.append([np.array(pro).mean(), fpr, th])
        # return [np.array(pro).mean(), fpr, th]

    @staticmethod
    def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3, mp=False):
        # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
        min_th, max_th = amaps.min(), amaps.max()
        delta = (max_th - min_th) / max_step
        pros, fprs, ths = [], [], []
        if mp:
            # multiprocessing.Process has no `get` function
            results = multiprocessing.Manager().list()
            jobs = []
            for th in np.arange(min_th, max_th, delta):
                job = multiprocessing.Process(target=Evaluator.cal_pro_thr, args=(results, th, amaps, masks,))
                job.start()
                jobs.append(job)
            for job in jobs:
                job.join()
            results = list(results)
            results.sort(key=lambda x: float(x[2]))
            for result in results:
                pros.append(result[0])
                fprs.append(result[1])
                ths.append(result[2])
        else:
            binary_amaps = np.zeros_like(amaps, dtype=np.bool_)
            for th in np.arange(min_th, max_th, delta):
                binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
                pro = []
                for binary_amap, mask in zip(binary_amaps, masks):
                    for region in measure.regionprops(measure.label(mask)):
                        tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                        pro.append(tp_pixels / region.area)
                inverse_masks = 1 - masks
                fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
                fpr = fp_pixels / inverse_masks.sum()
                pros.append(np.array(pro).mean())
                fprs.append(fpr)
                ths.append(th)
        pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
        idxes = fprs < expect_fpr
        fprs = fprs[idxes]
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
        return pro_auc


def get_evaluator(cfg_evaluator):
    evaluator, kwargs = Evaluator, cfg_evaluator.kwargs
    return evaluator(**kwargs)

