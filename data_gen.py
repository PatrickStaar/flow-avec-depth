import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random


def crawl_folders(folders_list, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        for folder in folders_list:
            intrinsics = np.genfromtxt(folder/'cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
            imgs = sorted(folder.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in range(-demi_length, demi_length + 1):
                    if j != 0:
                        sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        return sequence_set


def explore(folder_list, sequence_len):
    sequence=[]
    for f in folder_list:
        intrinsics= np.genfromtxt(f/'cam.txt', delimiter=',').astype(np.float32).reshape((3, 3))
        imgs=sorted(f.files('*.jpg'))

        for i in range(1,sequence_len):
            # select ref frame from previous 1 to 3 steps
            d = random.randint(1,3)
            if i-d < 0:
                d=1
            
            sample = {
                'intrinsics':intrinsics,
                'img_t1':imgs[i],
                'img_t0':imgs[i-d]
            }
            sequence.append(sample)
        random.shuffle(sequence)
    return sequence

# for TUM格式
def explore_tum(folder_list, batch_size=1, fixed_intrinsics=True, max_interval=2, intrinsic_file=None):

    samples = []
    entire_set=[]

    if fixed_intrinsics :
        intrinsics=read_intrinsics(intrinsic_file) # needed to be done

    for f in folder_list:
        with open(f/'rgb.txt') as img_list:
            for i in img_list:
                if '#' in i:
                    continue
                i = i.rstrip('\n')
                stamp, rgb = i.split(' ')
                entire_set.append({
                    'stamp':float(stamp),
                    'img_src':rgb,
                })

    sample_len=len(entire_set)
    
    # selecting pairs
    for i in range(len(entire_set)):
        interval = np.random.randint(1, max_interval)

        if i + interval > sample_len - 1:
            t0 = i - interval
            t1 = i
        #  避免采样到两个不同场景下的样本
        elif entire_set[i+interval]['stamp'] - entire_set[i]['stamp'] > 10.0:
            t0 = i - interval
            t1 = i
        else:
            t0 = i
            t1 = i + interval

        samples.append({
            't0': entire_set[t0]['stamp'],
            't1': entire_set[t1]['stamp'],
            'img_t0': entire_set[t0]['stamp'],
            'img_t1': entire_set[t1]['stamp'], 
            'intrinsics':entire_set[t0]['intrinsics']
        })


def load_as_float(path):
    return imread(path).astype(np.float32)


class data_generator(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.samples = explore(self.scenes, sequence_length)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.samples[index]
        img1 = load_as_float(sample['img_t1'])
        img0 = [load_as_float(ref_img) for ref_img in sample['img_t0']]
        
        if self.transform is not None:
            imgs, intrinsics = self.transform([img1, img0], np.copy(sample['intrinsics']))
            img1 = imgs[0]
            img0 = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return img1, img0, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
