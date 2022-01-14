from data_gen import Kitti
from transforms import *
from torch.utils.data import DataLoader
import time
import os

from net import Disp
from geometrics import flow_warp
from loss_disp import *
from tqdm import tqdm
from cfg_disp import config
from collections import defaultdict
from logger import get_logger


def get_time():
    T = time.strftime('%m.%d.%H.%M.%S', time.localtime())
    return T


def get_loader(**cfg):
    dataset = Kitti(**cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle'],
        pin_memory=cfg['pin_memory'],
        num_workers=cfg['workers']
    )
    return loader


def update_lr(optimizer, base_lr, max_iters,
              cur_iters, power=0.9):
    if optimizer is None:
        return 0
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def update(items, loss_dict):
    for k, v in loss_dict.items():
        if isinstance(v, float):
            continue
        else:
            items[k] += v.detach_().cpu().item()
    return items


def update_eval(items, loss_dict):
    for k, v in loss_dict.items():
        items[k] += v
    return items


def make_message(loss_dict, head=''):
    msg = head
    for k, v in loss_dict.items():
        msg += '{}:{:.6f},'.format(k, v)
    return msg


def train(net, dataloader, device, optimizer, cfg, rigid=False, net_D=None, optimizer_D=None, scheduler_D=None):
    net.train()
    loss_per_epoch = defaultdict(int)
    process = tqdm(enumerate(dataloader))
    for i, input_dict in process:
        left = input_dict['images'].squeeze_(1).to(device)
        right = input_dict['stereo'].squeeze_(1).to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        B, _, H, W = left.size()
        input_data=left if not cfg['use_right'] else torch.cat([left,right],dim=1)
        disp = net(input_data)

        left_warped, masks = [], []

        for d in range(len(disp)):
            if d > 0:
                disp[d] = upsample(disp[d], (H, W))
            y_matrix = torch.zeros_like(disp[d])
            mask = torch.ones_like(disp[d])
            flow = torch.cat([-disp[d]/2, y_matrix], dim=1)
            left_warped.append(flow_warp(right, flow))
            masks.append(flow_warp(mask, flow))

        pred = dict(
            disp=disp,
            left_warped=left_warped,
            mask=masks
        )

        target = dict(
            left=left,
            # right=right,
            intrinsics=intrinsics,
            intrinsics_inv=intrinsics_inv
        )

        loss_per_iter = summerize(pred, target, cfg)
        optimizer.zero_grad()
        loss_per_iter['loss_G'].backward()
        optimizer.step()

        loss_per_epoch = update(loss_per_epoch, loss_per_iter)
        process.set_description(make_message(loss_per_iter, '>'))

    # calc average loss per epoch
    for k, v in loss_per_epoch.items():
        loss_per_epoch[k] = v/len(dataloader)

    return loss_per_epoch


@torch.no_grad()
def eval(net, dataloader, device, cfg):
    loss_per_validation = defaultdict(int)
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    for i, input_dict in val_process:
        left = input_dict['images'][0].to(device)
        right = input_dict['stereo'][0].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        input_data=left if not cfg['use_right'] else torch.cat([left,right],dim=1)

        disp = net(input_data)[0]

        disp.squeeze_(dim=1)
        pred = dict(
            disp=1/(disp+1e-3),
        )
        target = dict(
            img_src=left,
            img_tgt=right,
            intrinsics=intrinsics,
            intrinsics_inv=intrinsics_inv,
            depth_gt=input_dict['depth_gt']
        )

        # 具体的validation loss计算的指标和输出的形式还需确定
        loss_per_iter = evaluate_depth(target, pred)
        val_process.set_description('Evaluating...')
        loss_per_validation = update_eval(loss_per_validation, loss_per_iter)

    for k, v in loss_per_validation.items():
        loss_per_validation[k] = v/len(dataloader)

    return loss_per_validation


if __name__ == "__main__":

    log = get_logger(config['log'])

    train_loader = get_loader(**config['data']['train'])
    val_loader = get_loader(**config['data']['val'])
    log.info('Data Loaded')

    SEED = time.time()
    SEED = 100000
    log.info(f'SEED: {SEED}')

    if config['device'] == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.manual_seed(SEED)
    else:
        device = torch.device('cpu')
        torch.manual_seed(SEED)
    log.info(f'Torch Device: {device}')

    date = time.strftime('%y.%m.%d')
    save_pth = config['save_pth']

    mask_gen = None
    if config['losses']['use_mask']:
        from davos import Mask
        mask_gen = Mask()

    net = Disp(input_channels=6 if config['losses']['use_right'] else 3)
    net.to(device)
    net.init_weights()

    if config['pretrained_weights']:
        net.load(config['pretrained_weights'])
        log.info(f'Load Pretrained: {config["pretrained_weights"]}')

    opt = torch.optim.Adam(net.parameters(), lr=config['lr'])
    log.info(f'Optimizer: Adam\nLR: {config["lr"]}')

    log.info(f'Train Samples:{len(train_loader)}')
    log.info(f'Val Samples:{len(val_loader)}')

    min_loss = 1000.
    min_val_loss = 1000.

    for epoch in range(1, config['max_epoch']+1):
        train_avg_loss = train(net, train_loader, device,
                               opt, config['losses'])
        log.info(make_message(train_avg_loss,
                 f'Epoch-{epoch} Training Loss >> {train_avg_loss["loss_G"]}'))

        lr = update_lr(opt, config['lr'], config['max_epoch'], epoch)
        log.info(f'LR updated to {lr}')

        eval_avg_loss = eval(net, val_loader, device, config['losses'])
        log.info(make_message(eval_avg_loss,
                 f'Epoch-{epoch} Validation Loss >> '))

        min_loss = train_avg_loss['loss_G']
        filename = f'{get_time()}_ep{epoch}.pt'
        torch.save(net.state_dict(), f=os.path.join(save_pth, filename))

        if eval_avg_loss['rmse'] < min_val_loss:
            min_val_loss = eval_avg_loss['rmse']
            torch.save(net.state_dict(), f=config['eval_weights'])
            log.info(f'Save epoch {epoch} as the best.')

    log.info('Training Done.')
