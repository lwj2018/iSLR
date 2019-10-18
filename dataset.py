import torch.utils.data as data

from PIL import Image
import torch
import torch.nn as nn
import numpy as np

import os
import os.path as osp

class VideoRecord(object):
    def __init__(self,row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class iSLR_Dataset(data.Dataset):
    
    def __init__(self, video_root, list_file,
                modality='RGB', transform=None):
        self.video_root = video_root
        self.list_file = list_file
        self.modality = modality
        self.transform = transform
        
        self._parse_list()

    def _load_image(self, directory, idx):
        path_list = os.listdir(osp.join(self.video_root,directory))
        path_list.sort()
        if self.modality == 'RGB':
            try: 
                return [Image.open(osp.join(self.video_root,directory,path_list[idx])).convert('RGB')]
            except Exception:
                print('error loading image:', osp.join(self.root_path, directory, path_list[idx]))
                return [Image.open(osp.join(self.root_path, directory, path_list[0])).convert('RGB')]
        
    def _parse_list(self):
        tmp = [x.strip().split('\t') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>4]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d'%(len(self.video_list)))

    def __getitem__(self, index):
        record = self.video_list[index]
        images = list()
        for i in range(0,record.num_frames,4):
            img = self._load_image(record.path, i)
            images.extend(img)

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)