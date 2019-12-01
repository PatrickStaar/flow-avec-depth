
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
import cv2


def get_time():
    T = time.strftime('%y.%m.%d-%H.%M.%S', time.localtime())
    return T.split('-')


# 数据预处理
t = Compose([
    ArrayToTensor(),
    # Normalize(mean=cfg.mean, std=cfg.std),
])

print('composed transform')

# 定义数据集
testset = data_generator(
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
    shuffle=True,
    pin_memory=True,  # 锁页内存
)


print('defined data_loader')
# 定义summary
# test_writer=SummaryWriter(cfg.log)
# output_writer = []

# 设置随机数种子
SEED = 0
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
print('set to test mode')
net.eval()

# 是否导入预训练
net.load_state_dict(torch.load(cfg.weight_for_test))

# 启动summary
global_steps = 0


for i, (img0, img1, intrinsics, intrinsics_inv) in enumerate(test_loader):
    print(i)
    global_steps += 1
    # calc loading time

    # add Varibles
    img0 = img0.cuda()
    img1 = img1.cuda()
    intrinsics = intrinsics.cuda()
    intrinsics_inv = intrinsics_inv.cuda()

    depth_maps, pose, flows = net([img1, img0])

    depth0 = [d[:, 0] for d in depth_maps]
    depth1 = [d[:, 1] for d in depth_maps]

    flow0 = flows[0]
    flow1 = -flow0

    # generate multi scale mask, including forward and backward masks

    forward_warped = flow_warp(
        img0, flow0).cpu().detach().numpy().squeeze(0)
    backward_warped = flow_warp(
        img1, flow1).cpu().detach().numpy().squeeze(0)

    forward_warped = forward_warped.transpose(1, 2, 0)*0.5+0.5
    backward_warped = backward_warped.transpose(1, 2, 0)*0.5+0.5
    forward_img = img1.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
    backward_img = img0.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5


    cv2.imwrite(cfg.test_tmp/'{}_forward_src.jpg'.format(i),
                    np.uint8(forward_img*255))
    cv2.imwrite(cfg.test_tmp/'{}_backward_src.jpg'.format(i),
                    np.uint8(backward_img*255))

    cv2.imwrite(cfg.test_tmp/'{}_forward.jpg'.format(i),
                    np.uint8(forward_warped*255))
    cv2.imwrite(cfg.test_tmp/'{}_backward.jpg'.format(i),
                    np.uint8(backward_warped*255))
