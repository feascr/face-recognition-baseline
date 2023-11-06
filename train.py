import time, argparse
from pathlib import Path
from shutil import copyfile
import logging
import yaml

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F 
from addict import Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from metrics import compute_topk, compute_frr_far, compute_eer, compute_accuracy_at_threshold, plot_frr_far
from utils import seed_everything, iteration_time, configure_logger, AverageMeter
from models import create_model
from data.datasets import create_dataset
from data.samplers import create_sampler


def train(
    train_iter,
    model,
    device,
    train_loss,
    optimizer,
    grad_scaler,
    use_amp,
):
    loss_meter = AverageMeter()
    top1_acc_meter = AverageMeter()
    top5_acc_meter = AverageMeter()
    model.train()
    for images, targets in tqdm(train_iter):
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        with amp.autocast(enabled=use_amp):
            logits = model(images, targets)
            loss = train_loss(logits, targets)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=5)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        with torch.no_grad():
            loss_meter.update(loss.item(), len(targets))
            top1, top5 = compute_topk(
                logits.detach().cpu().float(),
                targets.detach().cpu().float(),
                topk=(1,5)
            )
            top1_acc_meter.update(top1, len(targets))
            top5_acc_meter.update(top5, len(targets))
    return loss_meter, top1_acc_meter, top5_acc_meter

    
def evaluate(
    val_iter,
    model,
    device,
    val_loss,
):
    if val_loss is not None:
        loss_meter = AverageMeter()
    else:
        loss_meter = None
    model.eval()
    all_cos_logits = []
    all_l2_logits = []
    all_targets = []
    # with torch.no_grad():
    for images1, images2, targets in tqdm(val_iter):
        images1 = images1.to(device)
        images2 = images2.to(device)
        targets = targets.to(device)
        
        embeddings1 = model.forward_embeddings(images1)
        embeddings2 = model.forward_embeddings(images2)
        cos_logits = F.cosine_similarity(embeddings1, embeddings2)
        l2_logits = -F.pairwise_distance(embeddings1, embeddings2, p=2)
        if val_loss is not None:
            loss = loss(cos_logits, targets)
            loss_meter.update(loss.item(), len(targets))
        else:
            loss = float('nan')
        all_cos_logits.append(cos_logits.detach().cpu().numpy())
        all_l2_logits.append(l2_logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    all_cos_logits = np.concatenate(all_cos_logits)
    all_l2_logits = np.concatenate(all_l2_logits)
    all_targets = np.concatenate(all_targets)
    # cosine metrics
    cos_frr, cos_far, cos_thresholds = compute_frr_far(all_cos_logits, all_targets)
    cos_eer, cos_eer_th = compute_eer(all_cos_logits, all_targets)
    cos_acc = compute_accuracy_at_threshold(all_cos_logits, all_targets, threshold=0.0)
    cos_acc_eer = compute_accuracy_at_threshold(all_cos_logits, all_targets, threshold=cos_eer_th)
    cos_frr_far_plot = plot_frr_far(cos_thresholds, cos_frr, cos_far, cos_eer, cos_eer_th)
    cos_frr_far_plot = torch.tensor(cos_frr_far_plot.transpose(2, 0, 1))
    # l2 metrics
    l2_frr, l2_far, l2_thresholds = compute_frr_far(all_l2_logits, all_targets)
    l2_eer, l2_eer_th = compute_eer(all_l2_logits, all_targets)
    l2_acc_eer = compute_accuracy_at_threshold(all_l2_logits, all_targets, threshold=l2_eer_th)
    l2_frr_far_plot = plot_frr_far(l2_thresholds, l2_frr, l2_far, l2_eer, l2_eer_th)
    l2_frr_far_plot = torch.tensor(l2_frr_far_plot.transpose(2, 0, 1))
    return {
        'cosine': (cos_eer, cos_eer_th, cos_acc, cos_acc_eer, cos_frr_far_plot),
        'l2': (l2_eer, l2_eer_th, l2_acc_eer, l2_frr_far_plot)
    }

def fit_model(
    model,
    train_iter,
    val_iter,
    train_loss,
    val_loss,
    optimizer,
    grad_scaler,
    device,
    writer,
    save_dir,
    num_iterations,
    use_amp
):
    # TODO: Add saving each_n
    for iteration in range(1, num_iterations + 1):
        backbone_training_lr = optimizer.param_groups[0]["lr"]
        head_training_lr = optimizer.param_groups[1]["lr"]

        logging.info('Iteration: {:d} | Training:'.format(iteration))
        train_start_time = time.time()
        # Training loop
        loss_meter, top1_acc_meter, top5_acc_meter = train(
            train_iter,
            model,
            device,
            train_loss,
            optimizer,
            grad_scaler,
            use_amp
        )
        train_end_time = time.time()
        with torch.no_grad():
            tr_iteration_mins, tr_iteration_secs = iteration_time(
                train_start_time,
                train_end_time
            )

            logging.info('Iteration: {:d} | Validating:'.format(iteration))
            # Validation loop
            val_start_time = time.time()
            metrics = evaluate(
                val_iter,
                model,
                device,
                val_loss
            )
            val_end_time = time.time()
            val_iteration_mins, val_iteration_secs = iteration_time(
                val_start_time,
                val_end_time
            )

            iteration_mins, iteration_secs = iteration_time(
                train_start_time,
                val_end_time
            )

            writer.add_scalar('iter_time/train', tr_iteration_mins * 60 + tr_iteration_secs, iteration)
            writer.add_scalar('iter_time/val', val_iteration_mins * 60 + val_iteration_secs, iteration)
            writer.add_scalar('iter_time/all', iteration_mins * 60 + iteration_secs, iteration)
            
            writer.add_scalar('lr/backbone', backbone_training_lr, iteration)
            writer.add_scalar('lr/head', head_training_lr, iteration)
            
            writer.add_scalar('loss_last/train', loss_meter.val, iteration)
            writer.add_scalar('loss_iter_average/train', loss_meter.avg, iteration)
            writer.add_scalar('top1_acc_acerage/train', top1_acc_meter.avg, iteration)
            writer.add_scalar('top5_acc_acerage/train', top5_acc_meter.avg, iteration)
            
            writer.add_scalar('val_accuracy/cosine_th0', metrics['cosine'][2], iteration)
            writer.add_scalar('val_accuracy/cosine_eer_th', metrics['cosine'][3], iteration)
            writer.add_scalar('val_eer/cosine', metrics['cosine'][0], iteration)
            writer.add_scalar('val_eer_th/cosine', metrics['cosine'][1], iteration)
            writer.add_image('val_frr_far_plot/cosine', metrics['cosine'][4], iteration)

            writer.add_scalar('val_accuracy/l2_eer_th', metrics['l2'][2], iteration)
            writer.add_scalar('val_eer/l2', metrics['l2'][0], iteration)
            writer.add_scalar('val_eer_th/l2', metrics['l2'][1], iteration)
            writer.add_image('val_frr_far_plot/l2', metrics['l2'][3], iteration)
            
            logging.info('Saving model ...')
            torch.save(
                {
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 
                '{}/iteration_{:d}.pt'.format(save_dir, iteration)
            )

            # Iteration information
            content = '\nIteration: {:d} | Iteration Time: {:d}m {:d}s\n'.format(
                iteration, 
                iteration_mins, 
                iteration_secs
            )
            content += '\tTrain loss: {:.5f}'.format(loss_meter.avg)
            content += '\tTrain top1 acc: {:.5f}'.format(top1_acc_meter.avg)
            content += '\tTrain top5 acc: {:.5f}'.format(top5_acc_meter.avg)
            content += '\n'

            content += '\tVal cosine acc at 0.0 th: {:.5f}'.format(metrics['cosine'][2])
            content += '\tVal cosine eer (th: {:.4f}): {:.5f}'.format(metrics['cosine'][1], metrics['cosine'][0])
            content += '\n'
            content += '\tVal l2 acc at eer th: {:.5f}'.format(metrics['l2'][2])
            content += '\tVal l2 eer (th: {:.4f}): {:.5f}'.format(metrics['l2'][1], metrics['l2'][0])
            content += '\n'
            logging.info(content)
        

def _init_model(config, device):
    # build model
    model = create_model(
        config.model_name,
        config.backbone_name,
        config.head_name,
        config.backbone_kwargs,
        config.head_kwargs
    )
    model = model.to(device)
    logging.info('Model {} is built and loaded\n'.format(config.model_name))
    return model


def _init_preprocessors():
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.Blur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.5),
        A.OneOf(
            [
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.1,
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.5, 
                    contrast_limit=0.4,
                    p=1.0)
            ], p=0.8),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        A.Resize(32, 32),
        ToTensorV2()
    ])
    val_transforms = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        A.Resize(32, 32),
        ToTensorV2()
    ])
    return train_transforms, val_transforms


def _init_dataloaders(config, train_transforms, val_transforms, device):
    train_ds = create_dataset(
        config.data.train_dataset,
        config.data.train_json_path,
        train_transforms
    )
    val_ds = create_dataset(
        config.data.val_dataset,
        config.data.val_json_path,
        val_transforms
    )
    
    # build simple weight sampler based on dataset provided distribution
    sampler = create_sampler(
        config.data.train_sampler,
        initial_weights=train_ds.get_info_for_sampler(), 
        batch_size=config.data.train_batch_size, 
        batches_per_iteration=config.data.batches_per_iteration,
    )
    
    # build dataloaders
    pin_memory = True if 'cpu' not in '{}'.format(device) else False

    train_iterator = DataLoader(
        train_ds, 
        batch_size=config.data.train_batch_size, 
        sampler=sampler, 
        num_workers=config.num_workers, 
        drop_last=True, 
        pin_memory=pin_memory, 
        persistent_workers=False,
    )
    
    val_iterator = DataLoader(
        val_ds, 
        batch_size=config.data.val_batch_size, 
        num_workers=config.num_workers, 
        drop_last=False, 
        pin_memory=pin_memory, 
        persistent_workers=False,
    )

    logging.info('DataLoaders are built with pin_memory: {}\n'.format(pin_memory))

    return train_iterator, val_iterator
    
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument(
        '--config', 
        type=str,
        default='./train_config.yml', 
        help='path to config file'
    )
    parser.add_argument(
        '--name-suffix', 
        type=str, 
        help='additional suffix to output folder name'
    )
    parser.add_argument('--device', type=str, default=None, help='device name')
    args = parser.parse_args()

    # config file opening
    with open(args.config) as cfg_f:
        # load dict keys as class attributes
        config = Dict(yaml.safe_load(cfg_f))

    # TODO: move all until fit_model to init func
    # prepare folder for saving results
    if args.name_suffix:
        suffix_name = '_' + args.name_suffix
    else:
        suffix_name = ''
    save_path = config.save_path
    if save_path is not None:
        res_folder_suffix = (
            str(int(time.time())) + '_' + config.model_name + suffix_name
        )
        save_dir = Path(save_path, res_folder_suffix)
        save_dir.mkdir(exist_ok=True)
        
        train_config_path = save_dir / 'train_config.yml'
        _ = copyfile(args.config, train_config_path)
        
        configure_logger(str(Path(save_dir, 'training_log.txt')))
        logging.info('Results dir is made: {}\n'.format(save_dir))

        # git_commit_hash_path = Path(save_dir, 'git_commit_hash.txt')
        # with open(git_commit_hash_path, 'w') as out_file:
        #     commit_hash = subprocess.check_output(
        #         ['git', 'rev-parse', 'HEAD']
        #     ).strip().decode()
        #     out_file.write('{}\n'.format(commit_hash))

        save_dir = str(save_dir)
    else:
        save_dir = None
        logging.debug('Results will not be saved\n')

    # seed everything
    seed_everything(config.seed)

    # set device
    device_name = config.device if args.device is None else args.device
    device = torch.device(
        device_name if (
            torch.cuda.is_available() and 'cuda' in device_name
        ) else 'cpu'
    )
    logging.info('Initialized device: {}\n'.format(device))

    # build model
    model = _init_model(config, device)
    
    # build preprocessor
    # TODO: add feature for configuration
    # TODO: serialize and save with experiment info
    train_transforms, val_transforms = _init_preprocessors()
    
    # build ds
    # TODO: json optimize
    train_iter, val_iter = _init_dataloaders(
        config,
        train_transforms,
        val_transforms,
        device
    )

    # build loss
    train_loss = torch.nn.CrossEntropyLoss()

    # build optmizer
    optim_parameters = [
        {'params': model.backbone_parameters(), 'lr': config.optim.backbone_lr},
        {'params': model.head_parameters(), 'lr': config.optim.head_lr}
    ] 
    # TODO: try different optim
    optimizer = optim.Adam(
        optim_parameters, 
        lr=config.optim.common_lr,
        weight_decay=config.optim.weight_decay
    )
    
    # TODO: try scheduler
    scheduler = None
    logging.info('Optimizer is built\n')

    logging.info('AMP is {}enabled\n'.format('not ' if not config.use_amp else ''))
    grad_scaler = amp.GradScaler(enabled=config.use_amp)
    
    # init tb logs
    writer = SummaryWriter(log_dir=str(Path(
            config.tensorboard_log_dir, res_folder_suffix
        ))
    )

    # start main train loop
    fit_model(
        model,
        train_iter,
        val_iter,
        train_loss,
        None,
        optimizer,
        grad_scaler,
        device,
        writer,
        save_dir,
        num_iterations=config.number_of_iterations,
        use_amp=config.use_amp
    )