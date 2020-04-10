import random
import numpy as np
import torch
from skimage import io, transform
import cv2
from path import Path
from torch.utils.data import Dataset
from kitti_utils import *



class Kitti(Dataset):
    def __init__(self,root, sample_list, input_size, intrinsics, 
        train=True, sequence=(-1,0),transform=None,target_transform=None, 
        shuffle=True, with_depth=False, with_flow=False, with_pose=False,**kwargs):
        
        super(Kitti, self).__init__()

        self.root=root
        self.train=train
        self.sequence=sequence
        self.transform=transform
        self.target_transform=target_transform
        self.H, self.W=input_size
        self.cast={'l':2,'r':3}

        self.with_depth=with_depth
        self.with_flow=with_flow
        self.with_pose=with_pose

        self.files=self._read_from_txt(sample_list)
        self.intrinsics=self._set_intrinsics(intrinsics)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        folder, frame_id, side=self.files[index].strip('\n').split()
        # default collect_fn_ in dataloader do not allow None type value
        inputs={
            # 'intrinsics':None,
            # 'intrinsics_inv':None,
            # 'images':None,
            # 'depth_gt':None,
            # 'flow_gt':None,
            # 'pose_gt':None,
        }     
        imgs=[]
        frame_id=int(frame_id)
        # TODO: 读取连续帧的方法，同时使用多少帧，从txt文件中读取路径还是？文件的命名以及时间点的选择？
        for i in self.sequence:
            imgs.append(self.get_img(folder,frame_id+i,side))
        
        imgs, intrinsics = self.transform(imgs, np.copy(self.intrinsics))
        inputs['intrinsics']=intrinsics
        inputs['intrinsics_inv']=intrinsics.inverse()
        inputs['images']=imgs

        if self.with_depth:
            inputs['depth_gt']=self.get_depth(folder,frame_id,side)
        if self.with_flow:
            inputs['flow_gt']=self.get_flow()
        if self.with_pose:
            inputs['pose_gt']=self.get_pose()
            
        return inputs

    def get_img(self,scene,frame_id,side,mode='.png'):
        path=os.path.join(self.root,scene,'image_0{}/data'.format(self.cast[side]), '{:010d}{}'.format(frame_id, mode))
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
        depth_gt = transform.resize(
            depth_gt, (self.H,self.W), order=0, preserve_range=True, mode='constant')
        return torch.from_numpy(depth_gt)

    def get_flow(self,scene,frame_id,side):
        raise NotImplementedError

    def get_pose(self,scene,frame_id,side):
        raise NotImplementedError

    def _set_intrinsics(self,intrinsics):
        if intrinsics is not None:
            if isinstance(intrinsics,str):
                intrinsics= np.genfromtxt(intrinsics).astype(np.float32)
                intrinsics = intrinsics.reshape((3, 3))
            elif isinstance(intrinsics,np.float32):
                intrinsics=intrinsics.reshape((3, 3))
        else:
            intrinsics = np.array([
                [0.58, 0.   , 0.5],
                [0.  , 1.92 , 0.5],
                [0.   , 0.   ,  1., ],], dtype=np.float32)
        return intrinsics
            

    def _load_as_float(self,path):
        return io.imread(path).astype(np.float32)

    def _read_from_txt(self,file_name):
        path=os.path.join(self.root,file_name)
        with open(path,'r') as sample_list:
            return sample_list.readlines()
