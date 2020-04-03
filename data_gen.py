import torch
import torch.utils.data as data
import numpy as np
from scipy.misc import imread
from path import Path
import random
import cfg
from kitti_utils import *
import skimage

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

def explore_kitti(folder_list, sequence_len = 0, shuffle=True):
    sequences=[]
    for f in folder_list:
        if f == '':
            continue
        intrinsics= np.genfromtxt(f/'cam.txt').astype(np.float32)
        intrinsics = intrinsics.reshape((3, 3))
        f_mono=f/'image_2'
        imgs=sorted(f_mono.files('*.png'))
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




# class data_generator(data.Dataset):

#     def __init__(self, root, seed=None, train=True, sequence_length=0, 
#     transform=None, target_transform=None, format='', shuffle=True):
#         np.random.seed(seed)
#         random.seed(seed)
#         self.root = Path(root)
#         # 在dataset/下创建train.txt 和 val.txt，内容为要使用的scence文件夹
#         scene_list_path = self.root/'train.txt' if train else self.root/'val.txt' # just for now
#         self.scenes = [self.root/(folder.strip('\n')) for folder in open(scene_list_path) if len(folder)>1]

#         if format == 'tum':
#             self.samples = explore_tum(self.scenes, shuffle, train=train)
#         else:
#             self.samples = explore_kitti(
#                 self.scenes, shuffle=shuffle, sequence_len= sequence_length)
#         self.transform = transform

#     def __getitem__(self, index):
#         sample = self.samples[index]
#         img0 = self.load_as_float(sample['img_t0'])
#         img1 = load_as_float(sample['img_t1'])
        
#         imgs, intrinsics = self.transform([img0, img1], np.copy(sample['intrinsics']))
#         img0, img1 = imgs        
            
#         return img0, img1, intrinsics, intrinsics.inverse()

    # def __len__(self):
    #     return len(self.samples)


class Kitti(data.Dataset):
    def __init__(self,root, sample_list, input_size, intrinsics, 
        train=True, sequence=[-1,0],transform=None,target_transform=None, 
        shuffle=True, with_depth=False, with_flow=False, with_pose=False,**kwargs):
        super(Kitti, self).__init__()
        self.root=root
        self.train=train
        self.sequence=sequence
        self.transform=transform
        self.target_transform=target_transform
        # self.shuffle=shuffle
        self.W, self.H=input_size
        # self.intrinsics=intrinsics
        self.cast={'l':2,'r':3}
        self.with_depth=with_depth
        self.with_flow=with_flow
        self.with_pose=with_pose

        self.files=self.read_from_txt(sample_list)
        self.intrinsics=self.set_intrinsics(intrinsics)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        folder, frame_id, side=self.files[index].split()
        
        inputs={
            'intrinsics':None,
            'intrinsics_inv':None,
            'images':None,
            'depth_gt':None,
            'flow_gt':None,
            'pose_gt':None,
        }     
        imgs=[]
        # TODO: 读取连续帧的方法，同时使用多少帧，从txt文件中读取路径还是？文件的命名以及时间点的选择？
        for i in self.sequence:
            imgs.append(self.get_img(folder,frame_id))
        
        imgs, intrinsics = self.transform(imgs, np.copy(self.intrinsics))
        inputs['intrinsics']=intrinsics
        inputs['intrinsics_inv']=intrinsics.inverse()
        inputs['images']=imgs

        if self.with_depth:
            inputs['depth_gt']=self.get_depth()
        if self.with_flow:
            inputs['flow_gt']=self.get_flow()
        if self.with_pose:
            inputs['pose_gt']=self.get_pose()
            
        return inputs

    def get_img(self,scene,frame_id,side,mode='.jpg'):
        path=os.path.join(self.root,scene,self.cast[side], '{:010d}{}'.format(frame_id, mode))
        img=self._load_as_float(path)
        return img 

    def get_depth(self,scene,frame_id,side):
        date=scene.split('/',1)[0]   
        calid_dir = os.path.join(self.root, date)
        velo_file = os.path.join(
            self.root,
            scene,
            "velodyne_points/data/{:010d}.bin".format(int(frame_id)))
        depth_gt = generate_depth_map(calid_dir, velo_file, self.cast[side])
        depth_gt = skimage.transform.resize(
            depth_gt, (self.H,self.W), order=0, preserve_range=True, mode='constant')

        return depth_gt

    def get_flow(self,scene,frame_id,side):
        raise NotImplementedError

    def get_pose(self,scene,frame_id,side):
        raise NotImplementedError

    def set_intrinsics(self,intrinsics):
        if self.intrinsics is not None:
            if isinstance(intrinsics,str):
                intrinsics= np.genfromtxt(intrinsics).astype(np.float32)
                self.intrinsics = intrinsics.reshape((3, 3))
            elif isinstance(intrinsics,np.float32):
                self.intrinsics=intrinsics.reshape((3, 3))
        else:
            self.intrinsics = np.array([
                [0.58, 0   , 0.5],
                [0   , 1.92, 0.5],
                [0   , 0   , 1, ],], dtype=np.float32)

    def _load_as_float(self,path):
        return imread(path).astype(np.float32)

    def _read_from_txt(self,file_name):
        path=os.path.join(self.root,file_name)
        with open(path,'r') as sample_list:
            return sample_list.readlines()
