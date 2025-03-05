import os
import copy
import glob
import shutil
import datetime
import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup
from util.vis import vis_rgb_gt_amp
from thop import profile
import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad
from util.util import save_metric
import time
# from . import TRAINER


# @TRAINER.register_module
class BaseTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.master, self.logger, self.writer = cfg.master, cfg.logger, cfg.writer
        self.local_rank, self.rank, self.world_size = cfg.local_rank, cfg.rank, cfg.world_size
        log_msg(self.logger, '==> Running Trainer: {}'.format(cfg.trainer.name))
        # =========> dataset <=================================
        # cfg.logdir_train, cfg.logdir_test = f'{cfg.logdir}/show_train', f'{cfg.logdir}/show_test'
        # makedirs([cfg.logdir_train, cfg.logdir_test], exist_ok=True)
        log_msg(self.logger, "==> Loading training dataset: {}".format(cfg.train_data.type))
        log_msg(self.logger, "==> Loading testing dataset: {}".format(cfg.test_data.type))

        self.train_loader, self.test_loader, self.train_set, self.test_set = get_loader(cfg)
        for info in self.train_set.verbose_info:
            log_msg(self.logger, "==> Dataset info: {}".format(info))
        for info in self.test_set.verbose_info:
            log_msg(self.logger, "==> Dataset info: {}".format(info))

        cfg.train_data.train_size, cfg.test_data.test_size = len(self.train_loader), len(self.test_loader)
        cfg.train_data.train_length, cfg.test_data.test_length = self.train_loader.dataset.length, self.test_loader.dataset.length
        self.test_cls_names = self.test_loader.dataset.cls_names
        self.train_cls_names = self.train_loader.dataset.cls_names
        self.all_cls_names = self.test_loader.dataset.all_cls_names

        # =========> model <=================================
        log_msg(self.logger, '==> Using GPU: {} for Training'.format(list(range(cfg.world_size))))
        log_msg(self.logger, '==> Building model')

        if self.cfg.mode in ['test']:
            if not (cfg.model.kwargs['checkpoint_path'] or cfg.trainer.resume_dir):
                cfg.model.kwargs['checkpoint_path'] = f'{self.cfg.logdir}/{self.train_cls_names}_ckpt.pth'
                cfg.model.kwargs['strict'] = False
                log_msg(self.logger, f"==> Automatically Generate checkpoint: {cfg.model.kwargs['checkpoint_path'] }")

        self.net = get_model(cfg.model)
        self.net.to('cuda:{}'.format(cfg.local_rank))
        self.net.eval()
        log_msg(self.logger, f"==> Load checkpoint: {cfg.model.kwargs['checkpoint_path']}") if cfg.model.kwargs[
            'checkpoint_path'] else None
        # print_networks([self.net], torch.randn(self.cfg.fvcore_b, self.cfg.fvcore_c, self.cfg.size, self.cfg.size).cuda(), self.logger) if self.cfg.fvcore_is else None

        ### Others
        log_msg(self.logger, '==> Creating optimizer')

        self.optim = get_optim(cfg.optim.kwargs, self.net, lr=cfg.optim.lr)
        self.amp_autocast = get_autocast(cfg.trainer.scaler)
        self.loss_scaler = get_loss_scaler(cfg.trainer.scaler)
        self.loss_terms = get_loss_terms(cfg.loss.loss_terms, device='cuda:{}'.format(cfg.local_rank))


        self.scheduler = get_scheduler(cfg, self.optim)
        self.evaluator = get_evaluator(cfg.evaluator)
        self.metrics = self.evaluator.metrics

        cfg.trainer.metric_recorder = dict()
        for idx, cls_name in enumerate(self.test_cls_names):
            for metric in self.metrics:
                cfg.trainer.metric_recorder.update({f'{metric}_{cls_name}': []})
                if idx == len(self.test_cls_names) - 1 and len(self.test_cls_names) > 1:
                    cfg.trainer.metric_recorder.update({f'{metric}_Avg': []})
        self.metric_recorder = cfg.trainer.metric_recorder

        self.iter, self.epoch = cfg.trainer.iter, cfg.trainer.epoch
        self.iter_full, self.epoch_full = cfg.trainer.iter_full, cfg.trainer.epoch_full
        if cfg.trainer.resume_dir:
            state_dict = torch.load(cfg.model.kwargs['checkpoint_path'], map_location='cpu')
            self.optim.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.loss_scaler.load_state_dict(state_dict['scaler']) if self.loss_scaler else None
            self.cfg.task_start_time = get_timepc() - state_dict['total_time']

        tmp_dir = f'{cfg.trainer.checkpoint}/tmp'
        tem_i = 0
        while os.path.exists(f'{tmp_dir}/{tem_i}'):
            tem_i += 1
        self.tmp_dir = f'{tmp_dir}/{tem_i}'
        log_cfg(self.cfg)

    def reset(self, isTrain=True):
        self.net.train(mode=isTrain)
        self.log_terms, self.progress = get_log_terms(
            able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test),
            default_prefix=('Train' if isTrain else 'Test'))

    def scheduler_step(self, step):
        self.scheduler.step(step)
        update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)

    ### Some methods need additional processing before training and testing
    def pre_train(self):
        pass

    ### Some methods need additional processing before training and testing
    def pre_test(self):
        pass

    def set_input(self, inputs, train=True):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()

        self.bs = self.imgs.shape[0]

        ### Note these are necessary for visualizations
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']

    def forward(self, train=True):
        pass

    def compute_anomaly_scores(self):
        return np.ndarray(0.), np.ndarray(0.) # anomaly maps, anomaly scores

    def compute_loss(self, train=True) -> (torch.Tensor, dict):
        # return loss, loss_log
        return torch.Tensor(0.), dict()


    def backward_term(self, loss_term, optim):
        optim.zero_grad()
        if self.loss_scaler:
            self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(),
                             create_graph=self.cfg.loss.create_graph)
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            optim.step()

    def optimize_parameters(self):
        with self.amp_autocast():
            self.forward(train=True)
            loss, loss_log = self.compute_loss(train=True)
        self.backward_term(loss, self.optim)

        for k, v in loss_log.items():
            update_log_term(self.log_terms.get(k),
                            reduce_tensor(v, self.world_size).clone().detach().item(),
                            1, self.master)

    def _finish(self):
        log_msg(self.logger, 'finish training')
        self._save_metrics()


    def _save_metrics(self):
        ## reorganize metric recorder
        metric_last = dict()
        metric_best = dict()

        for idx, cls_name in enumerate(self.test_cls_names):
            metric_last[cls_name] = dict()
            metric_best[cls_name] = dict()

            for metric in self.metrics:
                metric_last[cls_name][metric] = self.metric_recorder[f'{metric}_{cls_name}'][-1]
                metric_best[cls_name][metric] = max(self.metric_recorder[f'{metric}_{cls_name}'])

        metric_last_csv_path = f'{self.cfg.logdir}/{self.cfg.model.name}_{self.cfg.test_data.name}_last.csv'
        metric_best_csv_path = f'{self.cfg.logdir}/{self.cfg.model.name}_{self.cfg.test_data.name}_best.csv'

        for idx, cls_name in enumerate(self.test_cls_names):
            save_metric(metric_last[cls_name], self.all_cls_names, cls_name, metric_last_csv_path)
            save_metric(metric_best[cls_name], self.all_cls_names, cls_name, metric_best_csv_path)

    def complexity_analysis(self, input_size):
        image = torch.rand((input_size), dtype=torch.float32).cuda()
        flops, params = profile(self.net, inputs=(image,))

        print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
        print(f"Parameters: {params / 1e6:.3f} M")

        self.net.eval()  # 设置为评估模式
        start_time = time.time()

        with torch.no_grad():  # 不计算梯度
            for _ in range(100):  # 进行 100 次推理以提高稳定性
                _ = self.net(image)

        end_time = time.time()

        # 计算总时间和 FPS
        total_time = end_time - start_time
        fps = 100 / total_time  # 每秒帧数
        print(f"FPS: {fps:.2f}")



    def train(self):
        self.reset(isTrain=True)
        self.pre_train()
        train_length = self.cfg.train_data.train_size
        train_loader = iter(self.train_loader)
        a = self.epoch
        b = self.epoch_full
        c = self.iter
        d = self.iter_full
        while self.epoch < self.epoch_full and self.iter < self.iter_full:
            self.scheduler_step(self.iter)
            # ---------- data ----------
            t1 = get_timepc()
            self.iter += 1
            train_data = next(train_loader)

            self.set_input(train_data, train=True)
            t2 = get_timepc()
            update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
            # ---------- optimization ----------

            self.optimize_parameters()
            t3 = get_timepc()
            update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
            update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
            # ---------- log ----------
            if self.master:
                if self.iter % self.cfg.logging.train_log_per == 0:
                    msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length,
                                                     self.iter_full / train_length), self.master, None)
                    log_msg(self.logger, msg)
                    if self.writer:
                        for k, v in self.log_terms.items():
                            self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
                        self.writer.flush()
            if self.iter % self.cfg.logging.train_reset_log_per == 0:
                self.reset(isTrain=True)
            # ---------- update train_loader ----------
            if self.iter % train_length == 0:
                self.epoch += 1
                self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None

                if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
                    if self.epoch + self.cfg.trainer.test_per_epoch > self.epoch_full:  # last epoch
                        vis = True
                    else:
                        vis = False
                    # print(f'vis: {vis}')
                    self.test(vis)
                else:
                    self.test_ghost()
                self.cfg.total_time = get_timepc() - self.cfg.task_start_time
                total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
                eta_time_str = str(
                    datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
                log_msg(self.logger,
                        f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
                self.save_checkpoint()
                self.reset(isTrain=True)
                train_loader = iter(self.train_loader)
        self._finish()

    @torch.no_grad()
    def test_ghost(self):
        for idx, cls_name in enumerate(self.test_cls_names):
            for metric in self.metrics:
                self.metric_recorder[f'{metric}_{cls_name}'].append(0)
                if idx == len(self.test_cls_names) - 1 and len(self.test_cls_names) > 1:
                    self.metric_recorder[f'{metric}_Avg'].append(0)

    def save_scores(self, results, N):

        anomaly_maps = copy.deepcopy(results['anomaly_maps'])
        imgs_masks = copy.deepcopy(results['imgs_masks'])

        # 将 anomaly_maps 和 imgs_masks 展平，便于操作
        anomaly_maps_flat = anomaly_maps.flatten()
        imgs_masks_flat = imgs_masks.flatten()

        non_nan_mask = ~np.isnan(anomaly_maps_flat)

        # 使用布尔掩码提取 anomaly_maps_flat 和 imgs_masks_flat 中对应的非 NaN 值
        anomaly_maps_flat = anomaly_maps_flat[non_nan_mask]
        imgs_masks_flat = imgs_masks_flat[non_nan_mask]

        # anomaly_maps_flat = (anomaly_maps_flat-np.min(anomaly_maps_flat))/(np.max(anomaly_maps_flat)-np.min(anomaly_maps_flat))
        # 正常像素的异常分值（imgs_masks > 0.5）
        normal_scores = anomaly_maps_flat[imgs_masks_flat < 0.5]

        # 异常像素的异常分值（imgs_masks <= 0.5）
        abnormal_scores = anomaly_maps_flat[imgs_masks_flat >= 0.5]

        # 分别从正常和异常分值中随机采样 N 个值

        sampled_normal_scores = np.random.choice(normal_scores, min(N, normal_scores.shape[0]), replace=False)
        sampled_abnormal_scores = np.random.choice(abnormal_scores, min(N, abnormal_scores.shape[0]), replace=False)

        normal_scores_path = f'{self.cfg.logdir}/{self.test_cls_names}_normal_scores.npy'
        abnormal_scores_path = f'{self.cfg.logdir}/{self.test_cls_names}_abnormal_scores.npy'

        np.save(normal_scores_path, sampled_normal_scores)
        np.save(abnormal_scores_path, sampled_abnormal_scores)

        print(f"Saved normal scores to {normal_scores_path}")
        print(f"Saved abnormal scores to {abnormal_scores_path}")

    @torch.no_grad()
    def test(self, vis=True):
        self.pre_test()

        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys, anomaly_scores, img_names = [], [], [], [], [], []
        batch_idx = 0
        test_length = self.cfg.test_data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)

            self.set_input(test_data, train=False)
            self.forward(train=False)

            # loss, loss_log = self.compute_loss(train=False)
            # for k, v in loss_log.items():
            #     update_log_term(self.log_terms.get(k),
            #                     reduce_tensor(v, self.world_size).clone().detach().item(),
            #                     1, self.master)

            anomaly_map, anomaly_score = self.compute_anomaly_scores()
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis and vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.cfg.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int),
                               anomaly_map, self.cfg.model.name, root_out)

            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            anomaly_scores.append(anomaly_score)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            # ---------- log ----------
            if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                log_msg(self.logger, msg)

        results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps,
                       cls_names=cls_names, anomalys=anomalys, anomaly_scores=anomaly_scores)

        results = {k: np.concatenate(v, axis=0) for k, v in results.items()}

        # self.save_scores(results, N=10000)

        msg = {}
        for idx, cls_name in enumerate(self.test_cls_names):
            metric_results = self.evaluator.run(results, cls_name, self.logger)
            msg['Name'] = msg.get('Name', [])
            msg['Name'].append(cls_name)
            avg_act = True if len(self.test_cls_names) > 1 and idx == len(self.test_cls_names) - 1 else False
            msg['Name'].append('Avg') if avg_act else None

            for metric in self.metrics:
                metric_result = metric_results[metric] * 100
                self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                if self.writer:
                    self.writer.add_scalar(f'Test/{metric}_{cls_name}', metric_result, self.iter)
                    self.writer.flush()
                max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                msg[metric] = msg.get(metric, [])
                msg[metric].append(metric_result)
                msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                if avg_act:
                    metric_result_avg = sum(msg[metric]) / len(msg[metric])
                    self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                    max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                    max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                    msg[metric].append(metric_result_avg)
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')

        msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
                                stralign="center", )
        log_msg(self.logger, f'\n{msg}')

    def save_checkpoint(self):
        if self.master:
            checkpoint_info = {'net': trans_state_dict(self.net.get_learnable_params(), dist=False),
                               'optimizer': self.optim.state_dict(),
                               'scheduler': self.scheduler.state_dict(),
                               'scaler': self.loss_scaler.state_dict() if self.loss_scaler else None,
                               'iter': self.iter,
                               'epoch': self.epoch,
                               'metric_recorder': self.metric_recorder,
                               'total_time': self.cfg.total_time}
            save_path = f'{self.cfg.logdir}/{self.train_cls_names}_ckpt.pth'
            torch.save(checkpoint_info, save_path)
            torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/{self.train_cls_names}_net.pth')
            if self.epoch % self.cfg.trainer.test_per_epoch == 0:
                torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/{self.train_cls_names}_net_{self.epoch}.pth')

    def run(self):
        log_msg(self.logger,
                f'==> Starting {self.cfg.mode}ing')
        if self.cfg.mode in ['train']:
            self.train()
        elif self.cfg.mode in ['test']:
            self.test()
        else:
            raise NotImplementedError

