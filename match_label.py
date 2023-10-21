# -*- coding: utf-8 -*-
"""
@Time : 2022/5/29 22:17
@Author : HZR
@File : match_label
"""
"""
To match labels in excel tables
"""
from PIL import Image
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

excel_path = r"E:\img\label"
data = pd.read_excel(os.path.join(excel_path, "label.xlsx"), sheet_name="label")
# data.columns
patientID = data["patientID"]  #Patient ID column
F_grade = data["grade"] #label column


data['Whether corresponding cases'] = 0
data['US images or not'] = 0


file_path = r'E:\00 Fibrosis_img\data'  #your imgs files path
save_path = r'E:\00 Fibrosis_img\match_label\raw'  #your imgs save path

without_label = []

for patient in tqdm(os.listdir(file_path)):
    print(patient)
    index = np.where(data['patientID'] == patient)
    if index[0].size > 0:
        data.loc[index[0], 'Whether corresponding cases'] = 1
        F_grade = str(data.iloc[index]['grade'].values[0])
        img_path_list = glob.glob(r'E:\00 Fibrosis_img\data\{}\*'.format(patient))
        for n, i in enumerate(img_path_list):
            img = Image.open(i)
            if not os.path.exists(os.path.join(save_path, F_grade, patient)):
                os.makedirs(os.path.join(save_path, F_grade, patient))
            F_save = os.path.join(save_path, F_grade, patient, '{}_{}.png'.format(patient, n))
            img.save(F_save)
            data.loc[index[0], 'US images or not'] = 1
    else:
        without_label.append(patient)

data.to_excel(os.path.join(save_path, "match labels.xlsx"), index=False)