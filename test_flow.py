
import cfg
from data_gen import flow_generator
from transforms import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from model import PDF
from geometrics import inverse_warp, flow_warp, pose2flow, mask_gen, pose_vec2mat
from losses import *
from tensorboardX import SummaryWriter
import cv2


def get_time():
    T = time.strftime('%y.%m.%d-%H.%M.%S', time.localtime())
    return T.split('-')

def flow_visualize(flow):

    mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    flow_norm = (flow[:,:,0]**2+flow[:,:,1]**2)**0.5
    flow_dire = np.arctan2(flow[:,:,0], flow[:,:,1])

    max_value = np.abs(flow[:,:,:2]).max()

    channel0= ang*180/np.pi
    channel1 = cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX)
    channel2 = flow[:,:,2]
    # channel1 = channel1.clip(0,1)
    colormap=np.stack([channel0,channel1,channel2], axis=-1)
    colormap=cv2.cvtColor(np.float32(colormap),cv2.COLOR_HSV2RGB)

    return colormap


def flow_write(filename, flow):

    if flow.shape[2]==2:
        flow = np.stack([flow,np.ones_like(flow[:,:,0])], axis=-1)

    flow = np.float64(flow)
    flow[:,:,:2] = (flow[:,:,:2]*64+2**15)#.clip(0,2**16-1)
    flow = np.uint16(flow)
    flow = cv2.cvtColor(flow,cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, flow)

# 数据预处理
t = Compose([
    Scale(384,1280),
    ArrayToTensor(),
    Normalize(mean=cfg.mean, std=cfg.std),
])

print('composed transform')

# 定义数据集
testset = flow_generator(
    root=cfg.dataset_path,
    transform=t,
    sequence_length=cfg.sequence_len,
    format=cfg.dataset,
    shuffle=False,
    train=False

)

# 定义生成器
test_loader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,  # 锁页内存
)


print('defined data_loader')
# 定义summary
# test_writer=SummaryWriter(cfg.log)
# output_writer = []

# 设置随机数种子
SEED = 0
# 设置GPU

if not torch.cuda.is_available():
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
print('set to test mode')
net.eval()

# 是否导入预训练
net.load_state_dict(torch.load(cfg.weight_for_test))

# 启动summary
global_steps = 0


for i, (img0, img1, intrinsics, intrinsics_inv) in enumerate(test_loader):
    if i==10:
        break
    print(i)
    global_steps += 1
    # calc loading time

    # add Varibles
    img0 = img0.to(device)
    img1 = img1.to(device)
    # intrinsics = intrinsics.to(device)
    # intrinsics_inv = intrinsics_inv.to(device)

    depth_maps, pose, flows = net([img1, img0])

    # depth0 = [d[:, 0] for d in depth_maps]
    # depth1 = [d[:, 1] for d in depth_maps]
    flow=flows[0]
    


    forward_warped = flow_warp(
        img0, flow).cpu().detach().numpy().squeeze(0)
    # backward_warped = flow_warp(
    #     img1, -flow).cpu().detach().numpy().squeeze(0)

    forward_warped = forward_warped.transpose(1, 2, 0)*0.5+0.5
    # backward_warped = backward_warped.transpose(1, 2, 0)*0.5+0.5
    forward_img = img1.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
    backward_img = img0.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5


    cv2.imwrite(cfg.test_tmp/'{}_forward_src.jpg'.format(i),
                    np.uint8(forward_img*255))
    cv2.imwrite(cfg.test_tmp/'{}_backward_src.jpg'.format(i),
                    np.uint8(backward_img*255))

    cv2.imwrite(cfg.test_tmp/'{}_forward.jpg'.format(i),
                    np.uint8(forward_warped*255))
    # cv2.imwrite(cfg.test_tmp/'{}_backward.jpg'.format(i),
    #                 np.uint8(backward_warped*255))


    ## save flow to png file
    f = flow.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    f[...,0]=f[...,0]*(f.shape[0]-1)/2
    f[...,1]=f[...,1]*(f.shape[1]-1)/2

    f = np.concatenate([f,np.ones((f.shape[0],f.shape[1],1))],axis=-1)

    # color_map=flow_visualize(f)
    flow_write(cfg.test_tmp/('{}_flow.png'.format(i)),f)



    # generate multi scale mask, including forward and backward masks


