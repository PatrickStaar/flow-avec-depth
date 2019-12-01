import torch
import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import cfg


def explore(folder_list, sequence_len = 0):
    sequences=[]
    for f in folder_list:
        if f == '':
            continue
        intrinsics= np.genfromtxt(f/'cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
        imgs=sorted(f.files('*.jpg'))
        n = sequence_len if sequence_len > 0 else len(imgs) 

        for i in range(1,n):
            # select ref frame from previous 1 to 3 steps
            d = random.randint(1,cfg.max_interval)
            if i-d < 0:
                d=1
            
            sample = {
                'intrinsics':intrinsics,
                'img_t1':imgs[i],
                'img_t0':imgs[i-d]
            }
            sequences.append(sample)
        random.shuffle(sequences)
    return sequences


# for TUM格式
def explore_tum(folder_list, fixed_intrinsics=True, shuffle=True, train=True):

    sequences = []
    index = 'rgb.txt' if train else 'test.txt'

    if fixed_intrinsics :
        intrinsics=np.genfromtxt(cfg.fixed_intrinsics, delimiter=' ').astype(np.float32).reshape((3, 3)) 
    
    for f in folder_list:
        if f == '':
            continue

        if not fixed_intrinsics:
            intrinsics=np.genfromtxt(f/'cam.txt', delimiter=' ').astype(np.float32).reshape((3, 3)) 
        
        with open(f/index) as filelist:
            imgs = [l.rstrip('\n') for l in filelist.readlines()]

        for i in range(1,len(imgs)):
            
            d = random.randint(1,cfg.max_interval)
            if i-d < 0:
                d=1

            _, img_t1 = imgs[i].split(' ')
            _, img_t0 = imgs[i-d].split(' ')

            sequences.append({
                'intrinsics':intrinsics,
                'img_t0':f/img_t0,
                'img_t1':f/img_t1
            })
        if shuffle:
            random.shuffle(sequences)
    return sequences


def load_as_float(path):
    return imread(path).astype(np.float32)


class data_generator(data.Dataset):

    def __init__(self, root, seed=None, train=True, sequence_length=3, 
    transform=None, target_transform=None, format='', shuffle=True):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        # 在dataset/下创建train.txt 和 val.txt，内容为要使用的scence文件夹
        scene_list_path = self.root/'train.txt' if train else self.root/'train.txt' # just for now
        self.scenes = [self.root/(folder.strip('\n')) for folder in open(scene_list_path)]

        if format == 'tum':
            self.samples = explore_tum(self.scenes, shuffle, train=train)
        else:
            self.samples = explore(self.scenes, sequence_length)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        img0 = load_as_float(sample['img_t0'])
        img1 = load_as_float(sample['img_t1'])
        
        imgs, intrinsics = self.transform([img0, img1], np.copy(sample['intrinsics']))
        img0, img1 = imgs        
            
        return img0, img1, intrinsics, intrinsics.inverse()

    def __len__(self):
        return len(self.samples)
