# -*- coding: utf-8 -*-
"""
@Time : 2022/5/29 23:22
@Author : HZR
@File : split_data
"""

"""
Partition data set
"""

import os
from shutil import rmtree
import shutil
import random
import glob

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def main():
    path = r"E:\00 Fibrosis_img\match_label"  #your imgs path
    origin_data_path = os.path.join(path, "raw")
    assert os.path.exists(origin_data_path), "path '{}' does not exist.".format(origin_data_path)
    data_class = [cla for cla in os.listdir(origin_data_path)
                    if os.path.isdir(os.path.join(origin_data_path, cla))]
    mk_file(os.path.join(path, "train"))
    mk_file(os.path.join(path, "val"))
    for cla in data_class:
        mk_file(os.path.join(path, "train", cla))
        mk_file(os.path.join(path, "val", cla))

    for cla in data_class:
        cla_path = os.path.join(origin_data_path, cla)
        patient_id = os.listdir(cla_path)
        patient_num = len(patient_id)
        img_num = len(glob.glob(r'E:\00 Fibrosis_img\match_label\raw\{}\*\*.png'.format(cla)))
        print(cla, "patient_num", patient_num, "img_num", img_num)
        random_val_sample(patient_num, 0.3, img_num, cla_path, str(cla), path)

def random_val_sample(patient_num, split_rate, img_num, cla_path, cla, path):
    for seed in range(patient_num):
        random.seed(seed)
        print(seed)
        sample_name_list = []
        val_img_num = 0
        val_index = random.sample(range(patient_num), k=int(patient_num*split_rate))
        for i in val_index:
            sample_name = os.listdir(os.path.join(cla_path))[i]
            sample_name_list.append(sample_name)
            val_img_num = val_img_num + len(glob.glob(os.path.join(cla_path, '{}\\*.png'.format(sample_name))))
        val_radio = val_img_num/img_num
        if val_radio >= 0.29 and val_radio <= 0.31: # We sample 30% of the patients
            print(val_radio)
            for i in os.listdir(os.path.join(cla_path)):
                if i in sample_name_list:
                    goto_path = os.path.join(path, "val", cla)
                    for img in os.listdir(os.path.join(cla_path, i)):
                        try:
                            src_file = os.path.join(cla_path, i, img)
                            shutil.copy(src_file, goto_path)
                            # print(src_file)
                        except OSError as e:
                            print("Error: %s - %s." % (e.filename, e.strerror))
                else:
                    goto_path = os.path.join(path, "train", cla)
                    for img in os.listdir(os.path.join(cla_path, i)):
                        try:
                            src_file = os.path.join(cla_path, i, img)
                            shutil.copy(src_file, goto_path)
                        except OSError as e:
                            print("Error: %s - %s." % (e.filename, e.strerror))
            break
        else:
            print("The picture does not meet the sampling proportion, sample again")
            continue
    print("{} val patient:{}, imgs:{}, imgs radio:{}, seed:{}".format(cla, int(patient_num*split_rate), img_num, val_radio, seed))
    with open(os.path.join(path, 'result.txt'), 'a') as f:
        f.write("""{} val patient:{}, imgs:{}, imgs radio:{}, seed:{}\n""".format(cla, int(patient_num*split_rate), img_num, val_radio, seed))

if __name__ == '__main__':
    main()
    print("processing done!")