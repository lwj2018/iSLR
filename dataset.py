import torch.utils.data as data

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import os
import os.path as osp

import matplotlib.pyplot as plt

class VideoRecord(object):
    def __init__(self,row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def skeleton_file(self):
        return self._data[1]

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])

class iSLR_Dataset(data.Dataset):
    
    def __init__(self, video_root, skeleton_root, list_file,
                modality='RGB', transform=None, 
                length=16):
        self.video_root = video_root
        self.skeleton_root = skeleton_root
        self.list_file = list_file
        self.modality = modality
        self.transform = transform
        self.length = length
        self.width = 1280
        self.height = 720
        
        self._parse_list()

    def _load_image(self, directory, idx):
        path_list = os.listdir(osp.join(self.video_root,directory))
        path_list.sort()
        image_name = osp.join(self.video_root,directory,path_list[idx])
        # if idx==0: print(image_name)
        if self.modality == 'RGB':
            try: 
                return [Image.open(image_name).convert('RGB')]
            except Exception:
                print('error loading image:', osp.join(self.video_root, directory, path_list[idx]))
                return [Image.open(osp.join(self.video_root, directory, path_list[0])).convert('RGB')]
        
    def _parse_list(self):
        tmp = [x.strip().split('\t') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[2])>4]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def get_sample_indices(self,num_frames):
        indices = np.linspace(1,num_frames-1,self.length).astype(int)
        return indices
    
    def _load_data(self, filename):
        filename = osp.join(self.skeleton_root, filename)
        f = open(filename,"r")
        content = f.readlines()
        mat = self.content_to_mat(content)
        return mat

    def content_to_mat(self,content):
        self.skeleton_index = [5,9,6,10]
        mat = []
        for record in content:
            skeleton = record.rstrip("\n").rstrip(" ").split(" ")
            skeleton = [int(x) for x in skeleton]
            skeleton = np.array(skeleton)
            shape = skeleton.size
            skeleton = np.reshape(skeleton,[shape//2,2])
            skeleton = skeleton[self.skeleton_index]
            mat.append(skeleton)
        # mat: T,N,D
        mat = np.array(mat)
        mat = mat.astype(np.float32)
        return mat   

    def __getitem__(self, index):
        record = self.video_list[index]
        images = list()
        mat = self._load_data(record.skeleton_file)
        num_frames = record.num_frames if record.num_frames<mat.shape[0]\
            else mat.shape[0]
        indices = self.get_sample_indices(num_frames)
        for i in indices:
            img = self._load_image(record.path, i)
            images.extend(img)
        mat = mat[indices,:,:]
        
        heat_maps = []
        for i in range(mat.shape[0]):
            heat_map =[]
            for j in range(mat.shape[1]):
                x,y = mat[i,j,:]
                z = self.generate_gaussian(x,y)
                # if j==0:
                #     plt.subplot(4,4,i+1)
                #     plt.imshow(z)
                heat_map.append(z)
            heat_map = np.stack(heat_map,0)
            heat_maps.append(heat_map)
        # plt.show()
        heat_maps = np.stack(heat_maps,0)
    
        process_data = self.transform(images)
        return process_data, heat_maps, record.label

    def __len__(self):
        return len(self.video_list)

    def generate_gaussian(self,x,y):
        class Distribution():
            def __init__(self,mu,Sigma):
                self.mu = mu
                self.sigma = Sigma

            def two_d_gaussian(self,x):
                mu = self.mu
                Sigma =self.sigma
                n = mu.shape[0]
                Sigma_det = np.linalg.det(Sigma)
                Sigma_inv = np.linalg.inv(Sigma)
                N = np.sqrt((2*np.pi)**n*Sigma_det)

                fac = np.einsum('...k,kl,...l->...',x-mu,Sigma_inv,x-mu)

                Z = np.exp(-fac/2)/N
                Z = (Z-Z.min())/(Z.max()-Z.min())
                Z = torch.Tensor(Z)
                return Z
        N = 16
        X = np.linspace(0,1,N)
        Y = np.linspace(0,1,N)
        X,Y = np.meshgrid(X,Y)
        x = x/self.width
        y = y/self.height
        mu = np.array([x,y])
        Sigma = np.array([[0.01,0],[0,0.01]])
        pos = np.empty(X.shape+(2,))
        pos[:,:,0]= X
        pos[:,:,1] = Y

        p2 = Distribution(mu,Sigma)
        Z = p2.two_d_gaussian(pos)
        return Z