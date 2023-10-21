# -*- coding: utf-8 -*-
"""
@Time : 2022/5/27 16:01
@Author : HZR
@File : Fibrosis_dataset
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from os.path import join
from glob import glob
import numpy as np
import skimage.io as io
from skimage.transform import resize
import sklearn.preprocessing as preprocessing
import pandas as pd


class Fibrosis_dataset(Dataset):
    def __init__(self, data_dir, transform=None, df_data=None):
        # data_dir = r'G:\01 multistate_Markov\SWEclaNO3\train'
        assert (os.path.exists(data_dir)), "data_dir:{} 不存在！".format(data_dir)

        self.data_dir = data_dir  #图片路径
        self._get_img_info()  #病人信息
        self.transform = transform
        self.df_data = df_data  #临床数据

    def __getitem__(self, index):
        """
        img_info 是一个列表，每个元素包含图像的路径、long格式的标签
        """
        fn, label, img_ID = self.img_info[index]  #图片路径，标签，病人ID
        # print(fn, label, index)
        patient_ID = str(img_ID.split("_")[0])

        us_img = Image.open(fn).convert('RGB')  #打开图片us_img
        if self.transform is not None:
            us_img = self.transform(us_img)

        # ----临床数据
        # print('id', patient_ID)
        if self.df_data is not None:
            # print('patient_ID',patient_ID)
            # patient_ID = '2022051836120184'
            # patient_index = df_data['检查号'] == int(patient_ID)
            # sum(patient_index)
            # df_data = pd.read_excel(r"D:\00 hllldata\01 repository\Fibrosis-multi-mode\dataset\clinic.xlsx", 0)
            patient_index = self.df_data['检查号'] == int(patient_ID)  #病人索引，在excel表中找出与patient_ID（图片路径里面的）对应的检查号
            # print(patient_index)
            clinic_feature = self.df_data[patient_index].iloc[:, 2:6].values.astype('float32')  #通过病人索引找到临床指标
            print(str(img_ID), self.df_data[patient_index].iloc[:, 2:6])
            # print(self.df_data[patient_index])
            clinic_feature = torch.tensor(clinic_feature)  #转换成张量
        return us_img, clinic_feature, label, img_ID

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)

    def _get_img_info(self):
        """
        Returns:
        self.img_info = [(路径，标签，检查号储存为列表), (……)]
        """
        sub_dir_ = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        # print(sub_dir_) #name:['0', '1']
        sub_dir = [os.path.join(self.data_dir, c) for c in sub_dir_]
        # print(sub_dir) #[''G:\\01 multistate_Markov\\SWEclaNO3\\train\\1'', '...']

        self.img_info = []
        #img_info提取出训练集所有图片的路径，标签，检查号储存为列表
        for c_dir in sub_dir:  #0,1路径
            # print(c_dir)
            # c_dir = r"G:\img_swe\纤维化图\220720预测弹性label\train\0"
            for patient in os.listdir(c_dir):
                patient_data = os.path.join(c_dir, patient)  #图片路径
                path_img = [(os.path.join(patient_data), int(patient_data.split('\\')[-2]), str(patient))]  # 标签文件用数字命名
                self.img_info.extend(path_img)
        return self.img_info