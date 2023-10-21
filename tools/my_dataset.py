# -*- coding: utf-8 -*-
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        assert (os.path.exists(data_dir)), "data_dir:{} 不存在！".format(data_dir)

        self.data_dir = data_dir
        self._get_img_info()
        self.transform = transform
    
    def __getitem__(self, index):
        fn, label, img_name = self.img_info[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)  #c,h,w
            # print('图片的shape', img.shape)
            
            # img = np.reshape(img, (112, -1, 112, 3))    
            # img = img.transpose(1, 0, 2, 3)
            
        # label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2)  #转成one-hot编码
        return img, label, img_name   #图片、标签、图片名称
    
    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)
    
    def _get_img_info(self):
        # sub_dir_ = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        # sub_dir = [os.path.join(data_dir, c) for c in sub_dir_]
        sub_dir_ = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]  #['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        sub_dir = [os.path.join(self.data_dir, c) for c in sub_dir_]  #0-9每个文件夹的路径

        self.img_info = []
        for c_dir in sub_dir:
            path_img = [(os.path.join(c_dir, i), int(os.path.basename(c_dir)), i) for i in os.listdir(c_dir) if
                        i.endswith("png")]
            self.img_info.extend(path_img)
            

