from dataset import Kitti
from transforms import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from net.model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen, pose_vec2mat
from losses import *
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from cfg_default import config


def get_time():
    T = time.strftime('%m.%d.%H.%M.%S', time.localtime())
    return T


def get_loader(cfg):
    dataset = Kitti(**cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle'],
        pin_memory=cfg['pin_memory'],
    )
    return loader


def summary_printer(loss_dict):
    pass


def train(net, dataloader, device, optimizer, train_loss, rigid=False, eps=1e-5):
    net.train()
    loss_per_epoch = 0
    eps=1e-5
    process = tqdm(enumerate(dataloader))
    for i, (img0, img1, intrinsics, intrinsics_inv) in process:
        # add Varibles
        img0 = img0.to(device)
        img1 = img1.to(device)
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        depth_maps, pose, flows = net([img0, img1])

        flows_backward = [-f for f in flows]
        depth_t0_multi_scale = [1./(d[:, 0]+eps) for d in depth_maps]
        depth_t1_multi_scale = [1./(d[:, 1]+eps) for d in depth_maps]

        # generate multi scale mask, including forward and backward masks
        # TODO: 实现方法待改进
        if not rigid:
            masks = multi_scale_mask(
                multi_scale=4, depth=(depth_t0_multi_scale, depth_t1_multi_scale),
                pose=pose, flow=(flows, flows_backward),
                intrinsics=intrinsics, intrinsics_inv=intrinsics_inv)
        else:
            # mask is not needed in full rigid scenes
            masks = None

        # deprecated losses
        # train_loss['depth_consistency'] = loss_depth_consistency(
        #     depth_t0_multi_scale, depth_t1_multi_scale, pose=pose, img_src=img0, img_tgt=img1,
        #     multi_scale=4, intrinsics=intrinsics, intrinsics_inv=intrinsics_inv, mask=masks)

        # train_loss['flow_consistency'] = loss_flow_consistency(
        #     flows, img0, img1, multi_scale=4)

        # smoothness
        # train_loss['depth_smoothness'] = loss_smoothness(depth_t0_multi_scale)+\
        #                                 loss_smoothness(depth_t1_multi_scale)

        # 这里还需要改进，输入的格式
        pred = dict(
            depthmap=depth_t0_multi_scale,
            flowmap=flows,
            pose=pose
        )
        target = dict(
            img_src=img0,
            img_tgt=img1,
            intrinsics=intrinsics,
            intrinsics_inv=intrinsics_inv
        )

        # train_loss['flow_smoothness'] = loss_smoothness(flows)

        train_loss.summerize(pred, target)
        # total_loss = loss_sum(train_loss)

        # TODO: 损失的分类显示字符串函数summary_printer需要修改
        process.set_description(summary_printer)

        optimizer.zero_grad()
        train_loss.loss_per_iter.backward()
        optimizer.step()
        # TODO: 学习率调度需要实现
        # scheduler.step(metrics=total_loss)

        # calc time per step
        loss_per_epoch += train_loss.loss_per_iter.to('cpu').item()

        # if (i+1) % cfg.steps == 0:
        #     # 每 cfg.steps 批次打印一次
        #     train_writer.add_scalar(
        #         'depth_consistency_loss', losses['depth_consistency'].item(
        #         ), global_steps
        #     )  # summary 参数不可以是torch tensor
        #     train_writer.add_scalar(
        #         'flow_consistency_loss', losses['flow_consistency'].item(
        #         ), global_steps
        #     )

        #     print('--epoch {} iter {} loss:{:.6f} '.format(epoch +
        #             1, i+1, total_loss.to('cpu').item(), ))

    # calc average loss per epoch
    loss_per_epoch /= len(dataloader)
    print('>> Epoch {}: Average Loss:{:.6f}'.format(epoch+1, loss_per_epoch))

    return loss_per_epoch


def eval(net, dataloader, device, val_loss):
    loss_per_validation = 0
    eps=1e-5
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    for i, (img0, img1, intrinsics, intrinsics_inv) in val_process:
        img0 = img0.to(device)
        img1 = img1.to(device)
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        depthmap, pose, flow = net([img0, img1])

        flow_backward = [-f for f in flow]

        depth0 = [1./(d[:, 0]+eps) for d in depthmap]
        depth1 = [1./(d[:, 1]+eps) for d in depthmap]

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
            depthmap=depth0,
            flowmap=flow,
            pose=pose
        )
        target = dict(
            img_src=img0,
            img_tgt=img1,
            intrinsics=intrinsics,
            intrinsics_inv=intrinsics_inv
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
    train_loader = get_loader(config['data']['train'])
    val_loader = DataLoader(config['data']['val'])
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
    net = PDF(config['model'])
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

    train_loss = Loss(config['losses'])
    val_loss = Loss(config['losses'], train=False)

    min_loss = 1000.
    min_val_loss = 1000.
    # 开始迭代
    try:
        for epoch in range(config['max_epoch']):
            # set to train mode
            train_avg_loss = train(net, train_loader, device, opt, train_loss)
            eval_avg_loss = eval(net, val_loader, device, val_loss)

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

    except:
        print('****Exception****', file=log)
        log.close()

    print('Training Done.')
