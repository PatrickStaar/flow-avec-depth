from dataset import Kitti
from transforms import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import os
from net.model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen, pose_vec2mat
from losses import *
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from cfg_default import config


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


def update(items, loss_dict):
    for k,v in loss_dict.items():
        if isinstance(v,float):
            continue
        if k in items.keys():
            items[k]+=float(v.to('cpu').item())
        else:
            items[k]=float(v.to('cpu').item())
    return items


def train(net, dataloader, device, optimizer, cfg, rigid=False):
    net.train()
    eps=1e-4
    loss_per_epoch={}
    for i, input_dict in tqdm(enumerate(dataloader)):
        # add Varibles
        img0 = input_dict['images'][0].to(device)
        img1 = input_dict['images'][1].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        depth_maps, pose, flows = net([img0, img1])
        
        depth_maps = [1./(d+eps) for d in depth_maps]
        depth_t1_multi_scale = [1./(d[:, 1]+eps) for d in depth_maps]
        # flows_backward = [-f for f in flows]

        # generate multi scale mask, including forward and backward masks
        # TODO: 实现方法待改进
        if not rigid:
            pass
            # masks = multi_scale_mask(
            #     multi_scale=4, depth=(depth_t0_multi_scale, depth_t1_multi_scale),
            #     pose=pose, flow=(flows, flows_backward),
            #     intrinsics=intrinsics, intrinsics_inv=intrinsics_inv)
        else:
            # mask is not needed in full rigid scenes
            masks = None

        # 这里还需要改进，输入的格式
        pred = dict(
            depthmap=depth_maps,
            flowmap=flows,
            pose=pose)
        target = dict(
            img_src=img0,
            img_tgt=img1,
            intrinsics=intrinsics,
            intrinsics_inv=intrinsics_inv)

        loss_dict=summerize(pred, target,cfg)
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()

        # for v in loss_dict.values():
        #     if isinstance(v,torch.Tensor):
        #         v.detach_()
        
        # del input_dict
        # TODO: 学习率调度需要实现
        # scheduler.step(metrics=total_loss)

        # calc time per step
        loss_per_epoch += loss_dict['loss'].to('cpu').item()
        loss_per_epoch=update(loss_per_epoch,loss_dict)
        
        # TODO: 损失的分类显示字符串函数summary_printer需要修改
        # process.set_description(train_loss.loss_per_iter.to('cpu').item())
    
    # calc average loss per epoch
    msg=''
    for k,v in loss_per_epoch.items():
        msg+= '{}:{:.6f},'.format(k,v/len(dataloader))
    print('>> Epoch {}:{}}'.format(epoch+1,msg))

    return loss_per_epoch


def eval(net, dataloader, device, val_loss):
    loss_per_validation = 0
    eps=1e-4
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    for i, input_dict in val_process:
        img0 = input_dict['images'][0].to(device)
        img1 = input_dict['images'][1].to(device)
        intrinsics = input_dict['intrinsics'].to(device)
        intrinsics_inv = input_dict['intrinsics_inv'].to(device)
        
        depthmap, pose, flow = net([img0, img1])
        depth = 1./(depth+eps)

        # depth1 = [1./(d[:, 1]+eps) for d in depthmap]
        # flow_backward = [-f for f in flow]

        # generate multi scale mask, including forward and backward masks
        # if not cfg.rigid:
        #     masks = multi_scale_mask(
        #         multi_scale=1, depth=(depth0, depth1),
        #         pose=pose, flow=(flow, flow_backward),
        #         intrinsics=intrinsics, intrinsics_inv=intrinsics_inv)
        # else:
        #     # mask is not needed in full rigid scenes
        #     masks = None

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
        loss_per_iter = val_loss.summerize(pred, target)
        val_process.set_description("evaluating..., ")
        loss_per_validation += loss_per_iter.to('cpu').item()

    loss_per_validation /= len(dataloader)
    # TODO: 验证集各项损失显示
    print('>> Average Validation Loss:{:.6f} '.format(loss_per_validation))
    return loss_per_validation


if __name__ == "__main__":

    # 定义生成器
    train_loader = get_loader(**config['data']['train'])
    val_loader = get_loader(**config['data']['val'])
    print('Data Loaded.')

    # TODO: 定义summary


    # 设置随机数种子
    SEED = time.time()

    # 设置GPU or CPU
    if config['device'] == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.manual_seed(SEED)
    else:
        device = torch.device('cpu')
        torch.manual_seed(SEED)
    print('Torch Device:', device)

    # 定义saver
    date = time.strftime('%y.%m.%d')
    save_pth = config['save_pth']
    print('load model')

    # 定义模型
    net = PDF(**config['model'])
    net.to(device)

    # 是否导入预训练
    if config['pretrain']:
        net.load_state_dict(torch.load(config['pretrained_weights']))
    else:
        net.init_weights()

    # 设置优化器
    weights = net.parameters()
    opt = torch.optim.Adam(weights, lr=config['lr'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
    log_dir = os.path.join(config['log'],(get_time()+'.txt'))
    # 启动summary
    global_steps = 0

    print('Train Samples:', len(train_loader))
    print('Val Samples:', len(val_loader))
    log = open(log_dir, 'a+')

    print('Train Samples', len(train_loader), file=log)
    print('Val Samples:', len(val_loader), file=log)


    min_loss = 1000.
    min_val_loss = 1000.
    # 开始迭代
    
    for epoch in range(config['max_epoch']):
        # set to train mode
        train_avg_loss = train(net, train_loader, device, opt, config['losses'])
        eval_avg_loss = eval(net, val_loader, device,config['losses'])

        if train_avg_loss < min_loss:
            min_loss = train_avg_loss
            filename = '{}_ep{}.pt'.format(get_time(), epoch+1)
            torch.save(net.state_dict(), f=save_pth/filename)

        if eval_avg_loss < min_val_loss:
            min_val_loss = eval_avg_loss
            filename = '{}_ep{}_val.pt'.format(get_time(), epoch+1)
            torch.save(net.state_dict(), f=save_pth/filename)

        print('EP {} training loss:{:.6f} min:{:.6f}'.format(
            epoch, train_avg_loss, min_loss), file=log)
        print('EP {} validation loss:{:.6f} min:{:.6f}'.format(
            epoch, eval_avg_loss, min_val_loss), file=log)

    log.close()

    # except:
    #     print('****Exception****', file=log)
    #     log.close()

    print('Training Done.')
