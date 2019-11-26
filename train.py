
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


def get_time():
    T=time.strftime('%y.%m.%d-%H.%M.%S',time.localtime())
    return T.split('-')


# 数据预处理
t = Compose([
    
    ArrayToTensor(),
    Normalize(mean=cfg.mean,std=cfg.std),
    
])

print('composed transform')

# 定义数据集
trainset=data_generator(root=cfg.dataset_path,
                          transform=t,
                          sequence_length=cfg.sequence_len,
                          format=cfg.dataset,
                          shuffle=False
)

# valset=data_generator(root=cfg.dataset_path,
#                         transform=t,
#                         train=False,
#                         sequence_length=cfg.sequence_len,
#                         format=cfg.dataset
# )
print('defined dataset')

# 定义生成器
train_loader=DataLoader(trainset,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        pin_memory=True, #锁页内存
)
# val_loader=DataLoader(valset,
#                         batch_size=cfg.batch_size,
#                         shuffle=True,
#                         pin_memory=True
# )

print('defined loader')
# 定义summary
train_writer=SummaryWriter(cfg.log)
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


print('load model')
# 定义模型
net = PDF(mode='train')
net.to(device)
print('set to train mode')
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

losses={}  
# 开始迭代
for epoch in range(cfg.max_epoch):
    tic=time.time()
    iters=0
    accumulated_loss=0
    for i, (img1, img0, intrinsics, intrinsics_inv) in enumerate(train_loader):
        print(i)
        global_steps+=1
        # calc loading time

        # add Varibles
        img1=Variable(img1.cuda())
        img0=Variable(img1.cuda())
        intrinsics=Variable(intrinsics.cuda())
        intrinsics_inv=Variable(intrinsics_inv.cuda())

        depth_maps, pose, flows=net([img1,img0])

        depth_t0_multi_scale = [d[:,0] for d in depth_maps]
        depth_t1_multi_scale = [d[:,1] for d in depth_maps]

        flow_t0_multi_scale = [f[:,:2] for f in flows]
        flow_t1_multi_scale = [f[:,2:] for f in flows]



        # generate multi scale mask, including forward and backward masks        
        masks = multi_scale_mask(
            multi_scale=4, depth = (depth_t0_multi_scale, depth_t1_multi_scale),
            pose= pose, flow=(flow_t0_multi_scale, flow_t1_multi_scale), 
            intrinsics = intrinsics, intrinsics_inv = intrinsics_inv        
        )

        # 2 major losses
        losses['depth_consistency'] = loss_depth_consistency(
            depth_t0_multi_scale, depth_t1_multi_scale, pose=pose, img_src=img0, img_tgt=img1, multi_scale=4,
            intrinsics=intrinsics, intrinsics_inv = intrinsics_inv, mask=masks
        )
        
        losses['flow_consistency'] = loss_flow_consistency(flow_t0_multi_scale,flow_t1_multi_scale,img0, img1,multi_scale=4)

        total_loss = loss_sum(losses)     

        print('loss calced, now backwards')
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        scheduler.step(metrics=total_loss)

        #calc time per step
    
        accumulated_loss += total_loss.to('cpu').item()
        iters+=1

        if global_steps % cfg.steps == 0:
            # 每 cfg.steps 批次打印一次
            train_writer.add_scalar('depth_consistency_loss', losses['depth_consistency'].item(), global_steps) # summary 参数不可以是torch tensor
            train_writer.add_scalar('flow_consistency_loss', losses['flow_consistency'].item(), global_steps)
               
            print('[epoch %d,  %5d iter] total loss: %.6f '%(epoch + 1, i + 1, total_loss.to('cpu'), ))
            accumulated_loss = 0.0
    
    # validate

    # for i, (img1, img0, intrinsics, intrinsics_inv) in enumerate(val_loader):



    interval = time.time()-tic
    avg_loss=accumulated_loss/iters

    print('epoch {}: time elapse:{} average_loss:{} '.format(epoch,interval,avg_loss))
    
    if epoch == 0:
        min_loss = avg_loss
        continue
    
    if avg_loss < min_loss:
        min_loss = avg_loss
        

            
    


    # calc average loss per epoch

    




            




