from transforms import *
from easydict import EasyDict as edict
from . import config as hrnet_config
from . import update_config


update_config(hrnet_config,'/work/davos/hrnet/davis_hrnet_w48.yaml')
data=edict()

data.davis=edict()
data.davis.root='/dataset/DAVIS'
data.davis.val_list='ImageSets/480p/val_list.txt'
data.davis.train_list='ImageSets/480p/train.txt'
data.davis.max_len=2

data.fbms=edict()
data.fbms.root='/dataset/FBMS59'
data.fbms.val_list='val_files.txt'
data.fbms.train_list='train_files.txt'
data.fbms.max_len=2

data.ytb=edict()
data.ytb.root='/dataset/youtube'
data.ytb.val_list='val.txt'
data.ytb.train_list='train_seq5_final.txt'
data.ytb.max_len=2

# Training configurations
train=edict()
train.device='gpu'
train.eps=1e-5
train.pretrain=None
train.etc=dict(
    mode='SV',
    fusion='conv',
    flow_branch_grad=True,
    iter_per_epoch=1000,
    weight_G_S=1.0,
    weight_G_T=1.0,
    weight_D_S=0.5,
    weight_D_T=0.5,
    G_k=1,
    D_k=1,
)

# Evaluation configurations
val=edict()
val.loader=edict()
val.loader.batch_size=1
val.loader.shuffle=False
val.loader.workers=16
val.loader.transforms=Compose([
    #Scale(384,384),
    ArrayToTensor(),
    # Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5]),
])
val.post_process= False
val.prop=False #True
val.dataset='fbms'
val.device='gpu'
val.eps=1e-5
val.pretrain=(
    # add flow supervision
    # 'checkpoints/03.13.21.06.30-ep5-en-t.pt',
    # 'checkpoints/03.11.06.21.05-ep20-de.pt',
    '/work/pretrain/davos_en_da_fbms.pt',
    '/work/pretrain/davos_de_da_fbms.pt',
)

val.output='/work/mask/da.onfbms.thre8'
val.visual=False
val.threshold=0.8
val.color='green'
val.flow_size=(512,512)

config=edict()
config.data=data
config.train=train
config.val=val
config.hrnet=hrnet_config
