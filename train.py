
import cfg
from data_gen import data_generator
from transforms import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen, pose_vec2mat
from losses import *
# from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_time():
    T = time.strftime('%m.%d.%H.%M.%S', time.localtime())
    return T


def train(net, dataloader, device, optimizer):
    net.train()
    acc_loss = 0
    
    process = tqdm(enumerate(dataloader))
    for i, (img0, img1, intrinsics, intrinsics_inv) in process:
        # add Varibles
        img0 = img0.to(device)
        img1 = img1.to(device)
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        depth_maps, pose, flows = net([img0, img1])

        flows_backward = [-f for f in flows]

        depth_t0_multi_scale = [1./(d[:, 0]+cfg.eps) for d in depth_maps]
        depth_t1_multi_scale = [1./(d[:, 1]+cfg.eps) for d in depth_maps]
        

        # generate multi scale mask, including forward and backward masks
        if not cfg.rigid:
            masks = multi_scale_mask(
                multi_scale=4, depth=(depth_t0_multi_scale, depth_t1_multi_scale),
                pose=pose, flow=(flows, flows_backward),
                intrinsics=intrinsics, intrinsics_inv=intrinsics_inv)
        else:
            # mask is not needed in full rigid scenes
            masks = None

        # 2 major losses
        train_loss['depth_consistency'] = loss_depth_consistency(
            depth_t0_multi_scale, depth_t1_multi_scale, pose=pose, img_src=img0, img_tgt=img1,
            multi_scale=4, intrinsics=intrinsics, intrinsics_inv=intrinsics_inv, mask=masks)

        train_loss['flow_consistency'] = loss_flow_consistency(
            flows, img0, img1, multi_scale=4)

        # smoothness
        train_loss['depth_smoothness'] = loss_smoothness(depth_t0_multi_scale)+\
                                        loss_smoothness(depth_t1_multi_scale)
        # train_loss['flow_smoothness'] = loss_smoothness(flows)

        total_loss = loss_sum(train_loss)
        process.set_description("Epoch {} Iter {} Loss:{:.6f} ".format(
            epoch+1, i+1, total_loss.to('cpu').item()))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # scheduler.step(metrics=total_loss)

        # calc time per step
        acc_loss += total_loss.to('cpu').item()

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
    avg_loss = acc_loss/len(dataloader)
    print('***************************************************************************')
    print('**** Epoch {}: Average Loss:{:.6f} ****'
          .format(epoch+1, avg_loss))
    print('***************************************************************************')
    return avg_loss


def eval(net, dataloader, device):
    val_acc_loss=0
    val_total_loss = 0
    net.eval()
    val_process = tqdm(enumerate(dataloader))
    for i, (img0, img1, intrinsics, intrinsics_inv) in val_process:
        img0 = img0.to(device)
        img1 = img1.to(device)
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        depthmap, pose, flow = net([img0, img1])

        flow_backward = [-f for f in flow]

        depth0 = [1./(d[:, 0]+cfg.eps) for d in depthmap]
        depth1 = [1./(d[:, 1]+cfg.eps) for d in depthmap]

        # generate multi scale mask, including forward and backward masks
        if not cfg.rigid:
            masks = multi_scale_mask(
                multi_scale=1, depth=(depth0, depth1),
                pose=pose, flow=(flow, flow_backward),
                intrinsics=intrinsics, intrinsics_inv=intrinsics_inv)
        else:
            # mask is not needed in full rigid scenes
            masks = None

        # 2 major losses
        val_loss['depth_consistency'] = loss_depth_consistency(
            depth0, depth1, pose=pose, img_src=img0, img_tgt=img1,
            multi_scale=0, intrinsics=intrinsics, intrinsics_inv=intrinsics_inv, mask=masks)

        val_loss['flow_consistency'] = loss_flow_consistency(
            flow, img0, img1, multi_scale=0)

        val_total_loss = loss_sum(val_loss)
        val_process.set_description(
            "Val-Sample {}, Current Loss:{:.8f}".format(i+1, val_total_loss.to('cpu').item()))

        val_acc_loss += val_total_loss.to('cpu').item()

    val_avg_loss = val_acc_loss/len(dataloader)
    print('---------------------------------------------------------------------------')
    print('------------------- Average Validation Loss:{:.6f} ------------------------'.format(val_avg_loss))
    print('---------------------------------------------------------------------------')

    return val_avg_loss
    


# 数据预处理
t = Compose([
    Scale(384,1280),
    RandomHorizontalFlip(),
    ArrayToTensor(),
    Normalize(mean=cfg.mean, std=cfg.std),
])

t_val = Compose([
    Scale(384,1280),
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
    pin_memory=False,  # 锁页内存
)

valset = data_generator(
    root=cfg.dataset_path,
    transform=t_val,
    train=False,
    sequence_length=cfg.sequence_len,
    format=cfg.dataset,
    shuffle=False,
)

val_loader = DataLoader(
    valset,
    batch_size=cfg.batch_size,
    shuffle=False,
    pin_memory=False
)


print('defined data_loader')
# 定义summary
# train_writer = SummaryWriter(cfg.log)
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


# 是否导入预训练
if cfg.pretrain:
    net.load_state_dict(torch.load(cfg.pretrained_weights))
else:
    net.init_weights()

# 设置优化器
weights = net.parameters()
opt = torch.optim.Adam(weights, lr=cfg.lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
logfile=cfg.log/(get_time()+'.txt')
# 启动summary
global_steps = 0
print('Batch Size:', cfg.batch_size)
print('Sample Number:', len(train_loader))
print('Val Sample Number:', len(val_loader))
with open(logfile,'w') as log:
    print('Batch Size:', cfg.batch_size,file=log)
    print('Sample Number:', len(train_loader),file=log)
    print('Val Sample Number:', len(val_loader),file=log)

train_loss = {}
val_loss = {}
# 开始迭代
try:
    log = open(logfile, 'w')
    for epoch in range(cfg.max_epoch):
        # set to train mode

        train_avg_loss = train(net, train_loader,device, opt)
        eval_avg_loss = eval(net,val_loader,device)
        
        if epoch == 0:
            min_loss = train_avg_loss
            min_val_loss = eval_avg_loss
            continue

        if train_avg_loss < min_loss:
            min_loss = train_avg_loss
            filename = '{}_ep{}.pt'.format(get_time(), epoch+1)
            torch.save(net.state_dict(), f=save_pth/filename)
        
        if eval_avg_loss < min_val_loss:
            min_val_loss = eval_avg_loss
            filename = '{}_ep{}_val.pt'.format(get_time(), epoch+1)
            torch.save(net.state_dict(), f=save_pth/filename)

        print('EP {} training loss:{:.6f} min:{:.6f}'.format(epoch,train_avg_loss,min_loss),file=log)
        print('EP {} validation loss:{:.6f} min:{:.6f}'.format(epoch,eval_avg_loss,min_val_loss),file=log)
    
    log.close()
        
except:
    print('****Exception****', file = log)
    log.close()

print('Training Finished')


