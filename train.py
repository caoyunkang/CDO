import argparse
from tqdm import tqdm
from datasets import *
from datasets.dataset import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *


def train_epoch(model: CDOModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str):
    # change the model into train mode
    model.train_mode()

    loss_sum = 0
    for (data, gt, _, _) in dataloader:
        data = data.to(device)
        outputs = model(data)
        loss = model.cal_loss(outputs['FE'], outputs['FA'], mask=gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum


def test_epoch(model: CDOModel, dataloader: DataLoader, device: str, is_vis, img_dir, class_name, cal_pro):
    # change the model into eval mode
    model.eval_mode()

    scores = None
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    for (data, mask, label, name) in dataloader:
        for d, n in zip(data, name):
            test_imgs.append(denormalization(d.cpu().numpy()))
            names.append(n)

        gt_list.extend(label.cpu().numpy())
        for i in range(mask.shape[0]):
            gt_mask_list.append(mask[i].squeeze().cpu().numpy())

        data = data.to(device)
        outputs = model(data)
        score = model.cal_am(**outputs)

        if scores is None:
            scores = []

        scores.extend(score)

    img_roc_auc, per_pixel_rocauc, pro_auc_score, threshold = \
        metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)

    if is_vis:
        plot_sample_cv2(names, test_imgs, {'CDO': scores}, gt_mask_list, save_folder=img_dir)
        plot_anomaly_score_distributions({'CDO': scores}, gt_mask_list, save_folder=img_dir,
                                         class_name=class_name)

    result_dict = {'i_roc': img_roc_auc * 100, 'p_roc': per_pixel_rocauc * 100, 'p_pro': pro_auc_score * 100,
                   'threshold': threshold}

    return result_dict


def main(args):
    kwargs = vars(args)

    logger.info('==========running parameters=============')
    for k, v in kwargs.items():
        logger.info(f'{k}: {v}')
    logger.info('=========================================')

    setup_seed(kwargs['seed'])
    device = f"cuda:{kwargs['gpu_id']}"
    kwargs['device'] = device

    # prepare the experiment dir
    model_dir, img_dir, tensorboard_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    h, w = test_dataset_inst.get_size()
    kwargs['out_size_h'] = h
    kwargs['out_size_w'] = w

    # get the model
    model = get_model_from_args(**kwargs)
    model = model.to(device)

    # get the tensorboard logger
    tensorboard_logger = get_tensorboard_logger_from_args(tensorboard_dir, True)

    if not kwargs['pure_test']:  # train the model first

        # get the optimizer
        optimizer = get_optimizer_from_args(model=model, weight_decay=0.0001, **kwargs)
        lr_schedule = get_lr_schedule(optimizer)

        # forward
        train_dataloader, train_dataset_inst = \
            get_dataloader_from_args(phase='train', perturbed=kwargs['MOM'], **kwargs)

        epoch_bar = tqdm(range(kwargs['num_epochs']), desc=f"CDO:{kwargs['class_name']}")

        for epoch in epoch_bar:

            loss_sum = train_epoch(model, train_dataloader, optimizer, device)
            tensorboard_logger.add_scalar('loss', loss_sum, epoch)

            if epoch % kwargs['validation_epoch'] == 0 or epoch == kwargs['num_epochs'] - 1:

                if epoch == kwargs['num_epochs'] - 1:
                    is_viz = kwargs['vis']
                else:
                    is_viz = False

                # as the pro metric calculation is costly, we only calculate it in the last evaluation
                metrics = test_epoch(model, test_dataloader, device, is_viz, img_dir,
                                     class_name=kwargs['class_name'], cal_pro=False)

                model_save_path = os.path.join(model_dir, f'{model_name}.pt')
                model.save(model_save_path, metrics)

                logger.info(f"\n")

                for k, v in metrics.items():
                    tensorboard_logger.add_scalar(f'{k}', v, epoch)
                    logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")

            lr_schedule.step()

    # directly utilize existing model for evaluation
    model_load_path = os.path.join(model_dir, f'{model_name}.pt')

    try:
        model.load(model_load_path)

        metrics = test_epoch(model, test_dataloader, device, True, img_dir,
                             class_name=kwargs['class_name'], cal_pro=kwargs['cal_pro'])
        logger.info(f"\n")

        for k, v in metrics.items():
            logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")

        # save in csv format
        save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                    kwargs['dataset'], csv_path)

    except:
        print(f'Evaluation error. Please check the existence of a trained model in {model_load_path}')


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec3d', choices=['mvtec2d', 'mvtec3d'])
    parser.add_argument('--class-name', type=str, default='bagel')
    parser.add_argument('--img-resize', type=int, default=256)
    parser.add_argument('--img-cropsize', type=int, default=256)

    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument("--validation-epoch", type=int, default=5)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument("--backbone", type=str, default="hrnet32",
                        choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'hrnet18', 'hrnet32',
                                 'hrnet48'])
    parser.add_argument("--MOM", type=str2bool, default=True)
    parser.add_argument("--OOM", type=str2bool, default=True)
    parser.add_argument("--gamma", type=float, default=2.)



    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
