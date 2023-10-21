# -*- coding: utf-8 -*-
"""
@Author : HZR
@File : train_FIBNet
"""
import sys
sys.path.append(r'G:\00 Fibrosis_img\01 FibNet code')
import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tools.my_dataset import Dataset
from tools.common_tools import ModelTrainer, show_confMat, plot_line
import timm
from PIL import Image
from tqdm import tqdm
# import albumentations as A
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd



BASE_DIR = r'E:\00 Fibrosis_img\match_label'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_train_model = r''
if __name__ == "__main__":
    # config
    train_dir = os.path.join(r'E:\00 Fibrosis_img\match_label\train')
    test_dir = os.path.join(r'E:\00 Fibrosis_img\match_label\val')
    
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print('creat result file')

    class_names = ('<8.7kPa', '>=8.7kPa')
    num_classes = 2
    MAX_EPOCH = 400
    BATCH_SIZE = 32
    LR = 0.001
    log_interval = 1
    val_interval = 1
    start_epoch = -1
    milestones = [10, 20, 40, 60, 80]  # divide it by 10 at 32k and 48k iterations

    # ============================ step 1/5 data ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    # norm_mean = [0.5, 0.5, 0.5]
    # norm_std = [0.5, 0.5, 0.5]

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  
        transforms.RandomCrop(256, padding=32),
        transforms.RandomGrayscale(p=0.5),
        transforms.ToTensor(),
        # transforms.RandomAffine(degrees=30),
        # transforms.RandomErasing(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # MyDataset
    train_data = Dataset(data_dir=train_dir, transform=train_transform)
    valid_data = Dataset(data_dir=test_dir, transform=valid_transform)
    # DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # ============================ step 2/5 model ============================
   

    class FIBNet(nn.Module):
         def __init__(self, class_num):
             super(FIBNet, self).__init__()
                        
             self.net = models.resnet152(pretrained=True)
             self.myfc = nn.Linear(self.net.fc.in_features, class_num)
             self.net.fc = nn.Identity()

         def forward(self, x):
             x = self.net(x)
             x = self.myfc(x)
             return x
    

    model = FIBNet(2)
    if pre_train_model:  
        model.load_state_dict(torch.load(os.path.join(pre_train_model))['model_state_dict']) 
        print('Load your own pre-training weights')
    model.to(device)
    # ============================ step 3/5 loss ============================
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1], dtype=torch.float).to(device))

    # ============================ step 4/5 optimizer ============================
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # ============================ step 5/5 train ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    auc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0
    best_loss = 10000
    best_auc = 0

    for epoch in range(start_epoch + 1, MAX_EPOCH):
        # break
        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train, auc_train = ModelTrainer.train(train_loader, model, criterion, optimizer, epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid, auc_valid = ModelTrainer.valid(valid_loader, model, criterion, device)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train AUC:{:.4f} Valid AUC:{:.4f} LR:{}".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, auc_train, auc_valid, optimizer.param_groups[0]["lr"]))
        with open(os.path.join(log_dir, 'log.log'), 'a') as f:
            f.write("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train AUC:{:.4f} Valid AUC:{:.4f} LR:{}\n".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, auc_train, auc_valid, optimizer.param_groups[0]["lr"]))

        scheduler.step()

        # Drawing
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        auc_rec["train"].append(auc_train), auc_rec["valid"].append(auc_valid)
        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="Accuracy", out_dir=log_dir)
        plot_line(plt_x, auc_rec["train"], plt_x, auc_rec["valid"], mode="AUC", out_dir=log_dir)

        # save results
        df_loss = pd.DataFrame(loss_rec)
        df_acc = pd.DataFrame(acc_rec)
        df_auc = pd.DataFrame(auc_rec)
        df_loss.to_csv(os.path.join(log_dir, "Loss.csv"),index=False)
        df_acc.to_csv(os.path.join(log_dir, "Accuracy.csv"),index=False)
        df_auc.to_csv(os.path.join(log_dir, "AUC.csv"),index=False)

        if best_auc < auc_valid:
            best_auc = auc_valid
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_loss": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best_auc.pkl")
            torch.save(checkpoint, path_checkpoint)

        if best_loss > loss_valid:
            best_loss = loss_valid
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_loss": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best_loss.pkl")
            torch.save(checkpoint, path_checkpoint)

        if best_acc < acc_valid:
            best_acc = acc_valid
            best_epoch = epoch
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

            # confMat
            show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH-1)
            show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH-1)


    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_acc, best_epoch+1))

    with open(os.path.join(log_dir, 'log.log'), 'a') as f:
        f.write(" done ~~~~ {}, best acc: {} in :{} epochs.\n ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                  best_acc, best_epoch+1))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
