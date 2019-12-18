#-*-coding=utf8-*-


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
from matplotlib import pyplot as plt


def get_time():
    T = time.strftime('%y.%m.%d-%H.%M.%S', time.localtime())
    return T.split('-')


t = Compose([
    Scale(384,1280),
    ArrayToTensor(),
    Normalize(mean=cfg.mean, std=cfg.std),
])
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

global_steps = 0
outcome = []

for i, (img0, img1, intrinsics, intrinsics_inv) in enumerate(test_loader):
    print(i)
    global_steps += 1
    # calc loading time

    # add Varibles
    img0 = img0.cuda()
    img1 = img1.cuda()
    intrinsics = intrinsics.cuda()
    intrinsics_inv = intrinsics_inv.cuda()

    depth_maps, pose, flows = net([img0, img1])

    # depth0 = [d[:, 0] for d in depth_maps]
    # depth1 = [d[:, 1] for d in depth_maps]

    flow0 = flows[0]

    # generate multi scale mask, including forward and backward masks

    forward_warped = flow_warp(
        img0, flow0).cpu().detach().numpy().squeeze(0)
    backward_warped = flow_warp(
        img1, -flow0).cpu().detach().numpy().squeeze(0)

    forward_warped = forward_warped.transpose(1, 2, 0)*0.5+0.5
    backward_warped = backward_warped.transpose(1, 2, 0)*0.5+0.5
    img1 = img1.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
    img0 = img0.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)*0.5+0.5
    
    outcome.append((forward_warped, backward_warped, img0, img1))

    img0 = cv2.cvtColor(img0,cv2.COLOR_RGB2BGR)
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    forward_warped = cv2.cvtColor(forward_warped,cv2.COLOR_RGB2BGR)

    cv2.imwrite(cfg.test_tmp/'{}_img00.jpg'.format(i),
                    np.uint8(img0*255))
    cv2.imwrite(cfg.test_tmp/'{}_img02.jpg'.format(i),
                    np.uint8(img1*255))

    cv2.imwrite(cfg.test_tmp/'{}_img01.jpg'.format(i),
                    np.uint8(forward_warped*255))
    # cv2.imwrite(cfg.test_tmp/'{}_backward.jpg'.format(i),
    #                 np.uint8(backward_warped*255))

# fig=plt.Figure()
# for warped, inverse_warped, img0, img1 in outcome:
#     i0=plt.subplot(221)
#     i0.set_title('warped')
#     i0.imshow(warped)

#     i1=plt.subplot(222)
#     i1.set_title('target')
#     i1.imshow(img1)

#     i2=plt.subplot(223)
#     i2.set_title('inverse_warped')
#     i2.imshow(inverse_warped)

#     i2=plt.subplot(224)
#     i2.set_title('source')
#     i2.imshow(img0)

#     plt.show()
