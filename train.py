
import cfg
from data_gen import data_generator
from transforms import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen, pose_vec2mat
from losses import *
from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_time():
    T = time.strftime('%m.%d.%H.%M.%S', time.localtime())
    return T


# 数据预处理
t = Compose([
    RandomHorizontalFlip(),
    ArrayToTensor(),
    Normalize(mean=cfg.mean, std=cfg.std),
])

print('composed transform')

# 定义数据集
trainset = data_generator(
    root=cfg.dataset_path,
    transform=t,
    sequence_length=cfg.sequence_len,
    format=cfg.dataset,
    shuffle=True
)

# 定义生成器
train_loader = DataLoader(
    trainset,
    batch_size=cfg.batch_size,
    shuffle=True,
    pin_memory=True,  # 锁页内存
)

# valset=data_generator(root=cfg.dataset_path,
#                         transform=t,
#                         train=False,
#                         sequence_length=cfg.sequence_len,
#                         format=cfg.dataset
# )

# val_loader=DataLoader(valset,
#                         batch_size=cfg.batch_size,
#                         shuffle=True,
#                         pin_memory=True
# )

print('defined data_loader')
# 定义summary
train_writer = SummaryWriter(cfg.log)
output_writer = []

# 设置随机数种子
SEED = time.time()

# 设置GPU
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(SEED)
else:
    device = torch.device('cpu')
    torch.manual_seed(SEED)

print('Torch Device:', device)

# 定义saver
date = time.strftime('%y.%m.%d')
save_pth = cfg.save_pth
print('load model')

# 定义模型
net = PDF()
net.to(device)
print('set to train mode')
net.train()

# 是否导入预训练
if cfg.pretrain:
    net.load_state_dict(torch.load(cfg.pretrained_weights))
else:
    net.init_weights()

# 设置优化器
weights = net.parameters()
opt = torch.optim.Adam(weights, lr=cfg.lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

# 启动summary
global_steps = 0
total_iteration = len(train_loader)
print('Sample Number:', total_iteration)

losses = {}
# 开始迭代
for epoch in range(cfg.max_epoch):
    tic = time.time()
    iters = 0
    accumulated_loss = 0
    process = tqdm(enumerate(train_loader))
    for i, (img0, img1, intrinsics, intrinsics_inv) in process:
        global_steps += 1

        # add Varibles
        img0 =img0.to(device)
        img1 =img1.to(device)
        intrinsics =intrinsics.to(device)
        intrinsics_inv =intrinsics_inv.to(device)

        depth_maps, pose, flows = net([img0, img1])
        
        flows_backward = [-f for f in flows]

        depth_t0_multi_scale = [d[:, 0] for d in depth_maps]
        depth_t1_multi_scale = [d[:, 1] for d in depth_maps]

        # flow_t0_multi_scale = [f[:, 0] for f in flows]
        # flow_t1_multi_scale = [f[:, 1] for f in flows]

        # generate multi scale mask, including forward and backward masks
        if not cfg.rigid:
            masks = multi_scale_mask(
                multi_scale=4, depth=(depth_t0_multi_scale, depth_t1_multi_scale),
                pose=pose, flow=(flows, flows_backward),
                intrinsics=intrinsics, intrinsics_inv=intrinsics_inv )
        else:
            # mask is not needed in full rigid scenes
            masks = None

        # 2 major losses
        losses['depth_consistency'] = loss_depth_consistency(
            depth_t0_multi_scale, depth_t1_multi_scale, pose=pose, img_src=img0, img_tgt=img1,
            multi_scale=4, intrinsics=intrinsics, intrinsics_inv=intrinsics_inv, mask=masks )

        losses['flow_consistency'] = loss_flow_consistency(
            flows, img0, img1, multi_scale=4 )

        total_loss = loss_sum(losses)
        process.set_description("Epoch {}, Iter {}, Current Loss:{:.8f} ".format(epoch+1, i+1, total_loss.to('cpu').item()))

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        # scheduler.step(metrics=total_loss)

        # calc time per step
        accumulated_loss += total_loss.to('cpu').item()


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

    # validate

    # for i, (img1, img0, intrinsics, intrinsics_inv) in enumerate(val_loader):

    # calc average loss per epoch
    interval = time.time()-tic
    avg_loss = accumulated_loss/float(total_iteration)
    print('***************************************************************************')
    print('**** Epoch {}: Time Elapse:{:.4f} Iteration:{} Average Loss:{:.6f} ****'\
        .format(epoch+1, interval, total_iteration, avg_loss))
    print('***************************************************************************')
    
    if epoch == 0:
        min_loss = avg_loss
        continue

    if avg_loss < min_loss:
        min_loss = avg_loss
        filename = '{}_epoch_{}.pt'.format(get_time(), epoch+1)
        torch.save(net.state_dict(), f=save_pth/filename)
