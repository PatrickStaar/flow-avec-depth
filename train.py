import cfg
from data_gen import data_generator
from transforms import * 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen
from losses import *

def get_time():
    T=time.strftime('%y.%m.%d-%H.%M.%S',time.localtime())
    return T.split('-')

# 数据预处理
t = Compose([
    RandomHorizontalFlip(),
    ArrayToTensor(),
    Normalize(mean=cfg.mean,std=cfg.std)
])

# 定义数据集
trainset=data_generator(root='./datasets/kitti/train',
                          transform=t,
                          sequence_length=cfg.sequence_len
)

valset=data_generator(root='./datasets/kitti/val',
                        transform=t,
                        train=False,
                        sequence_length=cfg.sequence_len
)

# 定义生成器
train_loader=DataLoader(trainset,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        pin_memory=True, #锁页内存
)
val_loader=DataLoader(valset,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        pin_memory=True
)


# 定义summary

# 定义saver
date=time.strftime('%y.%m.%d')
save_pth = cfg.save_pth

# 定义模型
net = PDF(mode='train')

# 是否导入预训练
if cfg.pretrain:
    net.load_state_dict(torch.load(cfg.pretrained_weights))
if cfg.from_checkpoint:
    pass
    # net.load_state_dict(torch.load(cfg.))
else:
    net.init_weights()

# 设置优化器
weights = net.parameters()
opt = torch.optim.Adam(weights, lr = cfg.lr)

# 设置gpu


# 启动summary

# 开始迭代
for epoch in range(cfg.max_epoch):

    for i, (img1, img0, intrinsics, intrinsics_inv) in enumerate(trainset):
        # calc loading time

        # add Varibles
        img1=Variable(img1.cuda())
        img0=Variable(img1.cuda())
        intrinsics=Variable(intrinsics.cuda())
        intrinsics_inv=Variable(intrinsics_inv.cuda())

        depth_maps, pose, flows=net([img1,img0])

        losses=[]

        for j in range(len(depth_maps)):
            # 尺寸问题待解决
            img1_warped_d = inverse_warp(img0, ,pose,intrinsics,intrinsics_inv)
            img0_warped_d = inverse_warp(img1, depth_maps[j][0],pose,intrinsics,intrinsics_inv)

            img1_warped_f = flow_warp(img0, flows[j][1])
            img0_warped_f = flow_warp(img1, flows[j][0])

            # pose inv is needed

            flow_rigid_foward = pose2flow(depth_maps[j][1], pose, intrinsics, intrinsics_inv)
            flow_rigid_backward = pose2flow(depth_maps[j][0], pose_inv, intrinsics, intrinsics_inv)
            
            occulsion_mask0=mask_gen(flows[j][1],flow_rigid_foward)
            occulsion_mask1=mask_gen(flows[j][0],flow_rigid_backward)

            # 损失计算
            losses.append(
                loss_depth_consistency(depth_maps[j][1], depth_maps[0],
                                    pose=pose,img_src=img0,img_tgt=img1,
                                    intrinsics,intrinsics_inv)) # ATTENTION masks are missing 
            losses.append(
                loss_flow_consistency(flows[j][1],flows[j][0],img0, img1)
            )





            




