'''
data layout:
3drender
    --- subfoler (i.e. category)
        --- uid_folder
            --- {00x}.png, cameras.npz
'''

import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import math


class Zero123plusData(Dataset):
    def __init__(self,
        root_dir='3drender/',
        meta_fname='',
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.validation = validation

        self.cate_folders = []             # subfolers containing the folers of multiple objects, e.g., rendering_zero123plus_0-10
        self.uid_folders = []              # folders named with the ids of 3D object, e.g., xxx/000.png

        meta_uid_path = os.path.join(root_dir, meta_fname)
        if os.path.isfile(meta_uid_path):
            if meta_uid_path.endswith('.json'):         # lvis
                with open(meta_uid_path, 'r') as f:
                    lvis_dict = json.load(f)
                uids = []
                for k in lvis_dict.keys():
                    uids.extend(lvis_dict[k])
                self.uid_folders = uids
            else:
                with open(meta_uid_path, 'r') as f:     # clean uid txt file
                    uids = f.read().split('\n')
                self.uid_folders = [uid for uid in uids if len(uid) > 0]

            for obj_id in self.uid_folders:
                for idx in range(0, 160, 10):
                    cur_image_folder = 'rendering_zero123plus{}-{}'.format(idx, idx+10)
                    #cur_obj_ids = os.listdir(os.path.join(self.root_dir, cur_image_folder))
                    if os.path.exists(os.path.join(self.root_dir, cur_image_folder,obj_id)):
                        self.cate_folders.append(cur_image_folder)
                        break
            assert len(self.cate_folders) == len(self.uid_folders)
            
        else:
            cates = [item for item in os.listdir(root_dir) if not item.endswith('.txt') and not item.endswith('.json')]
            for cate in cates:
                cur_uids = os.listdir(os.path.join(root_dir, cate))
                self.uid_folders += cur_uids
                self.cate_folders += [cate] * len(cur_uids)
                    
            assert len(self.cate_folders) == len(self.uid_folders)
        
        if self.validation:
            self.uid_folders = self.uid_folders[-16:] # used last 16 as validation
            self.cate_folders = self.cate_folders[-16:] # used last 16 as validation
        else:
            self.uid_folders = self.uid_folders[:-16]
            self.cate_folders = self.cate_folders[:-16]
            
        print('============= length of current splitted dataset %d =============' % len(self.uid_folders))

    def __len__(self):
        return len(self.uid_folders)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T
        
    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def __getitem__(self, index):
        while True:
            image_path = os.path.join(self.root_dir, self.cate_folders[index], self.uid_folders[index])

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            pose_list = []
            try:
                input_cameras = np.load(os.path.join(image_path, 'cameras.npz'))['cam_poses']
                for idx in range(7):
                    img, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bkg_color)
                    img_list.append(img)

                    pose = input_cameras[idx]
                    pose_list.append(pose)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.uid_folders))
                continue

            break
        
        imgs = torch.stack(img_list, dim=0).float()                                    # (7, 3, H, W)
        assert len(pose_list) == 7
        Ts = [[self.get_T(cond_RT=pose_list[i], target_RT=pose_list[j]) if j > i else torch.tensor(0) for j in range(7)] for i in range(5)]

        data = {
            'images': imgs,
            'Ts': Ts,
            'uid': self.uid_folders[index]
        }
        return data
        
   


