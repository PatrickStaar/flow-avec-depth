import cfg
from data_gen import data_generator
from transforms import * 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen
from losses import *
from tensorboardX import SummaryWriter


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
train_writer=SummaryWriter(cfg.save_pth)
output_writer = []

# 设置随机数种子
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 设置GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 定义saver
date=time.strftime('%y.%m.%d')
save_pth = cfg.save_pth

# 定义模型
net = PDF(mode='train')
net.train()

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

# 设置gpu


# 启动summary
global_steps=0
accumulated_loss=0
# 开始迭代
for epoch in range(cfg.max_epoch):
    tic=time.time()
    for i, (img1, img0, intrinsics, intrinsics_inv) in enumerate(train_loader):
        global_steps+=1
        # calc loading time

        # add Varibles
        img1=Variable(img1.cuda())
        img0=Variable(img1.cuda())
        intrinsics=Variable(intrinsics.cuda())
        intrinsics_inv=Variable(intrinsics_inv.cuda())

        depth_maps, pose, flows=net([img1,img0])

        # generate multi scale mask, including forward and backward masks        
        masks = multi_scale_mask(
            multi_scale=4, depth = depth_maps, pose= pose, flow=flows, 
            intrinsics = intrinsics, intrinsics_inv = intrinsics_inv        
        )

        # 2 major losses
        losses['depth_consistency'] = loss_depth_consistency(
            depth_maps[1], depth_maps[0], pose=pose, img_src=img0, img_tgt=img1, multi_scale=4,
            intrinsics=intrinsics, intrinsics_inv = intrinsics_inv, mask=masks
        )
        losses={}  
        losses['flow_consistency'] = loss_flow_consistency(flows[1],flows[0],img0, img1,multi_scale=4)

        total_loss = final_loss(losses)     

        opt.zero_grad()
        total_loss.backward()
        opt.step()
        scheduler.step(metrics=total_loss)

        #calc time per step
    
        accumulated_loss += total_loss.to('cpu').item()

        if global_steps % cfg.steps == 0:
            # 每 cfg.steps 批次打印一次
            train_writer.add_scalar('depth_consistency_loss', losses['depth_consistency'].item(), global_steps) # summary 参数不可以是torch tensor
            train_writer.add_scalar('flow_consistency_loss', losses['flow_consistency'].item(), global_steps)
               
            print('[%d, %5d] loss: %.6f elapse: %.4f'%(epoch + 1, i + 1, accumulated_loss / cfg.steps, interval))
            accumulated_loss = 0.0
    
    # validate

    # for i, (img1, img0, intrinsics, intrinsics_inv) in enumerate(val_loader):



    interval = time.time()-tic
    print('epoch {}: time elapse:{}'.format(epoch,interval))

    # calc average loss per epoch

    




            




