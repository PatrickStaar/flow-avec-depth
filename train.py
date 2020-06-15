from dataset import Kitti
from transforms import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
from net.model import PDF
from net.discriminator import DCGAN_Discriminator
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen, pose_vec2mat
from losses import *
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from cfg_default import config
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
    )
    return loader


def lr_update(optimizer):
    pass

def update(items, loss_dict):
    for k, v in loss_dict.items():
        if isinstance(v, float):
            continue
        else:
            items[k] += v.detach_().cpu().item()
    return items


def make_message(loss_dict, head=''):
    msg = head
    for k, v in loss_dict.items():
        msg += '{}:{:.6f},'.format(k, v)
    return msg


def train(net, dataloader, device, optimizer, cfg, rigid=False, net_D=None, optimizer_D=None):
    net.train()
    eps = 1e-4
    loss_per_epoch = defaultdict(int)
    process = tqdm(enumerate(dataloader))
    for i, input_dict in process:
        # add Varibles
        img0 = input_dict['images'][0].to(device)
        img1 = input_dict['images'][1].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        B, _, H, W = img0.size()

        # set discriminator
        for param in net_D.parameters():
            param.requires_grad = False

        depth_maps, pose, flows = net([img0, img1])

        pred = dict([])
        target = dict(
            img_src=img0,
            img_tgt=img1,
            intrinsics=intrinsics,
            intrinsics_inv=intrinsics_inv)

        # if cfg['use_depth']:
        if depth_maps is not None:
            depth_maps = [upsample(1/(d*cfg['depth_scale']+cfg['depth_eps']),(H,W)) for d in depth_maps]

        rigid_flow = [pose2flow(d.squeeze_(dim=1), pose,intrinsics, intrinsics_inv)
                        for d in depth_maps]
        depth_warped = [flow_warp(img0, f) for f in rigid_flow]

        depth_disc_score = net_D(depth_warped[0])

        pred['depth_map'] = depth_maps
        pred['depth_warped'] = depth_warped
        pred['depth_disc_score'] = depth_disc_score

        # if cfg['use_flow']:
        flows = [upsample(f,(H,W)) for f in flows]
        flow_warped = [flow_warp(img0, upsample(f, (H, W))) for f in flows]

        flow_disc_score = net_D(flow_warped[0])

        pred['flow_map'] = flows
        pred['flow_warped'] = flow_warped
        pred['flow_disc_score'] = flow_disc_score
    
        mask = mask_gen(rigid_flow[0], flows[0]) if cfg['use_mask'] else None
        pred['mask'] = mask
        
        loss_per_iter = summerize(pred, target, cfg)
        optimizer.zero_grad()
        loss_per_iter['loss_G'].backward()
        optimizer.step()

        # bring back discriminator
        
        optimizer_D.zero_grad()
        for param in net_D.parameters():
            param.requires_grad = True

        target_D = net_D(img0)
        loss_per_iter['loss_D_pos'] = loss_disc(target_D, torch.ones_like(target_D))
        loss_per_iter['loss_D_pos'].backward()

        depth_warped[0].detach_()
        depth_D = net_D(depth_warped[0])
        loss_per_iter['loss_D_depth']=0.5*loss_disc(depth_D, torch.zeros_like(depth_D))
        loss_per_iter['loss_D_depth'].backward()

        flow_warped[0].detach_()
        flow_D = net_D(flow_warped[0])
        loss_per_iter['loss_D_flow']= 0.5*loss_disc(flow_D, torch.zeros_like(flow_D))
        loss_per_iter['loss_D_flow'].backward()

        optimizer_D.step()

        # TODO: 学习率调度需要实现
        # scheduler.step(metrics=total_loss)

        loss_per_epoch = update(loss_per_epoch, loss_per_iter)

        process.set_description(make_message(loss_per_iter, '>'))

    # calc average loss per epoch
    for k, v in loss_per_epoch.items():
        loss_per_epoch[k] = v/len(dataloader)

    return loss_per_epoch


def eval(net, dataloader, device, cfg):
    loss_per_validation = defaultdict(int)
    eps = 1e-4
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    for i, input_dict in val_process:
        img0 = input_dict['images'][0].to(device)
        img1 = input_dict['images'][1].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)

        depth, pose, flow = net([img0, img1])
        depth = 1/(depth*cfg['depth_scale']+cfg['depth_eps'])

        pred = dict(
            depthmap=depth,
            flowmap=flow,
            pose=pose
        )
        target = dict(
            img_src=img0,
            img_tgt=img1,
            intrinsics=intrinsics,
            intrinsics_inv=intrinsics_inv,
            depth_gt=input_dict['depth_gt']
        )

        # 具体的validation loss计算的指标和输出的形式还需确定
        loss_per_iter = evaluate(target, pred, cfg['weights'])
        val_process.set_description("evaluating..., ")
        loss_per_validation = update(loss_per_validation, loss_per_iter)

    for k, v in loss_per_validation.items():
        loss_per_validation[k] = v/len(dataloader)

    return loss_per_validation


if __name__ == "__main__":

    # 定义生成器
    train_loader = get_loader(**config['data']['train'])
    val_loader = get_loader(**config['data']['val'])
    print('Data Loaded.')

    # 设置随机数种子
    SEED = time.time()
    SEED = 100000

    # 设置GPU or CPU
    if config['device'] == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.manual_seed(SEED)
    else:
        device = torch.device('cpu')
        torch.manual_seed(SEED)
    print('Torch Device:', device)

    # TODO 替换掉其他代码中的cuda()

    # 定义saver
    date = time.strftime('%y.%m.%d')
    save_pth = config['save_pth']
    print('load model')

    # 定义generator
    net = PDF(**config['model'])
    net.to(device)
    net.init_weights()

    # 是否导入预训练
    if config['pretrain']:
        net.load_state_dict(torch.load(
            config['pretrained_weights']), strict=False)

    # 定义discriminator
    net_D = DCGAN_Discriminator(n_channel=3)
    net_D.to(device)
    x = net_D.parameters()
    # 设置优化器
    opt = torch.optim.Adam(net.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2, factor=0.5, min_lr=1e-7, cooldown=1)

    opt_D = torch.optim.Adam(net_D.parameters(), lr=config['lr_D'])
    # TODO 定义判别器scheduler

    log = get_logger(config['log'])

    log.info('Train Samples:{}'.format(len(train_loader)))
    log.info('Val Samples:{}'.format(len(val_loader)))

    min_loss = 1000.
    min_val_loss = 1000.
    # 开始迭代
    # try:

    for epoch in range(config['max_epoch']):
        # set to train mode
        train_avg_loss = train(net, train_loader, device, opt,
                               config['losses'], net_D=net_D, optimizer_D=opt_D)  # net_D不写默认为不用判别器
        log.info(make_message(train_avg_loss, 'Epoch-{} Training Loss >>'.format(epoch+1)))

        # TODO 这里需要对G,D分别调节学习率

        scheduler.step(train_avg_loss['loss_G'])

        eval_avg_loss = eval(net, val_loader, device, config['losses'])
        log.info(make_message(eval_avg_loss, 'Epoch-{} Validation Loss >>'.format(epoch+1)))

        # TODO 保存判别器模型
        if train_avg_loss['loss_G'] < min_loss:
            min_loss = train_avg_loss['loss_G']
            filename = '{}_ep{}.pt'.format(get_time(), epoch+1)
            torch.save(net.state_dict(), f=os.path.join(save_pth, filename))

        elif eval_avg_loss < min_val_loss:
            min_val_loss = eval_avg_loss
            filename = '{}_ep{}_val.pt'.format(get_time(), epoch+1)
            torch.save(net.state_dict(), f=os.path.join(save_pth, filename))
        else:
            pass

    log.info('Training Done.')

    # except BaseException as ex:
    # log.error(str(ex))
