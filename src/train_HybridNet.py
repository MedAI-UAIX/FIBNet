# -*- coding: utf-8 -*-
"""
@Author : HZR
@File : train_HybridNet
"""
import os
import numpy as np
import torch.nn as nn
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sklearn.preprocessing as preprocessing
from datetime import datetime
from Fibrosis_dataset import Fibrosis_dataset
from tools.common_tools import ModelTrainer, show_confMat, plot_line
import pandas as pd

# config
BASE_DIR = r'E:\00 Fibrosis_img\match_label'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre_train_model = False

# data_dir
dataset_dir = os.path.join(BASE_DIR, "dataset")
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
clinic_dir = r'E:\00 Fibrosis_img\match_label\dataset\clinic.xlsx'

# config
class_names = ('<8.7kPa', '>=8.7kPa')
num_classes = 2
MAX_EPOCH = 200  # 182     # 64000 / (45000 / 128) = 182 epochs
BATCH_SIZE = 16
LR = 0.001
log_interval = 1
val_interval = 1
start_epoch = -1
milestones = [10, 20, 40, 60, 80]  # divide it by 10 at 32k and 48k iterations

# ============================ step 1/5 data ============================
# transform
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
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

# df_clinic
def scaler_for_minmax(clinic_dir, dataset):
    df_temp = pd.read_excel(clinic_dir, dataset)
    Y = df_temp.iloc[:, :2]  # patientID+label
    X = df_temp.iloc[:, 2:6]  # clinical data
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max = min_max_scaler.fit_transform(X)
    X_min_max = pd.DataFrame(min_max, columns=X.columns.values)
    df_XY = pd.concat([Y, X_min_max], axis=1)
    return df_XY

df_train = scaler_for_minmax(clinic_dir, "train")
df_val = scaler_for_minmax(clinic_dir, "val")

# MyDataset
train_data = Fibrosis_dataset(data_dir=train_dir, transform=train_transform, df_data=df_train)
valid_data = Fibrosis_dataset(data_dir=val_dir, transform=valid_transform, df_data=df_val)

# DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  ##us_img, clinic_feature, label, patient_ID
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 model ============================
class FIBNet(nn.Module):
    def __init__(self):
        super(FIBNet, self).__init__()

        self.net = models.resnet152(pretrained=True)
        # self.myfc = nn.Linear(self.net.fc.in_features, 512)  #第一次
        self.myfc = nn.Linear(self.net.fc.in_features, 64)
        self.net.fc = nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.myfc(x)
        return x


class ClinicNet(nn.Module):
    def __init__(self):
        super(ClinicNet, self).__init__()
        # self.fc_1 = nn.Linear(4, 128)  #第一次
        # self.fc_2 = nn.Linear(128, 64)
        # print(self)
        self.fc_1 = nn.Linear(4, 16)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.fc_2 = nn.Linear(16, 32)
        # self.fc_3 = nn.Linear(64, 32)
        # self.fc_4 = nn.Linear(32, class_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # print("x.view", x)
        x = self.fc_1(x)  # !!!
        # print("fc_1(x)", x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x = self.fc_3(x)
        # x = self.relu(x)
        # x = self.fc_4(x)
        return x


class HybridNet(nn.Module):
    def __init__(self, class_num=2):
        super(HybridNet, self).__init__()
        self.FIB_net = FIBNet()
        self.clinic_net = ClinicNet()
        self.fc_1 = nn.Linear(64 + 32, 32)
        # self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(32, class_num)
        # self.fc_2 = nn.Linear(128, class_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, us, clinic):
        # print('clinic', clinic.shape)
        x_img = self.FIB_net(us)
        # print('x_img', x_img.shape)
        x_clinic = self.clinic_net(clinic)
        # print('x_clinic', x_clinic.shape)
        x = torch.concat([x_img, x_clinic], dim=1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


model = HybridNet()

if pre_train_model:
    model.load_state_dict(torch.load(os.path.join(pre_train_model))['model_state_dict'])
    print('Load the pre-training weight')
model.to(device)
# ============================ step 3/5 loss ============================
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1], dtype=torch.float).to(device))

# ============================ step 4/5 optimizer ============================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# ============================ step 5/5 train ============================
if __name__ == "__main__":
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    auc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0
    best_loss = 10000
    best_auc = 0
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print('create results~~~~')

    for epoch in range(start_epoch + 1, MAX_EPOCH):
        # epoch = 0
        # break
        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train, auc_train = ModelTrainer.train(train_loader, model, criterion, optimizer,
                                                                         epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid, auc_valid = ModelTrainer.valid(valid_loader, model, criterion, device)
        print(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train AUC:{:.4f} Valid AUC:{:.4f} LR:{}".format(
                epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, auc_train, auc_valid,
                optimizer.param_groups[0]["lr"]))
        with open(os.path.join(log_dir, 'log.log'), 'a') as f:
            f.write(
                "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train AUC:{:.4f} Valid AUC:{:.4f} LR:{}\n".format(
                    epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, auc_train, auc_valid,
                    optimizer.param_groups[0]["lr"]))

        scheduler.step()

        # plot loss acc auc
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        auc_rec["train"].append(auc_train), auc_rec["valid"].append(auc_valid)
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="Accuracy", out_dir=log_dir)
        plot_line(plt_x, auc_rec["train"], plt_x, auc_rec["valid"], mode="AUC", out_dir=log_dir)

        # saving results
        df_loss = pd.DataFrame(loss_rec)
        df_acc = pd.DataFrame(acc_rec)
        df_auc = pd.DataFrame(auc_rec)
        df_loss.to_csv(os.path.join(log_dir, "Loss.csv"), index=False)
        df_acc.to_csv(os.path.join(log_dir, "Accuracy.csv"), index=False)
        df_auc.to_csv(os.path.join(log_dir, "AUC.csv"), index=False)

        #saving the best weight
        if best_auc < auc_valid:
            best_auc = auc_valid
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_loss": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best_auc.pkl")
            torch.save(checkpoint, path_checkpoint)
            print('saving best AUC weight')

        if best_loss > loss_valid:
            best_loss = loss_valid
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_loss": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best_loss.pkl")
            torch.save(checkpoint, path_checkpoint)
            print('saving minimum loss weight')

        if best_acc < acc_valid:
            best_acc = acc_valid
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
            print('saving best ACC weight')
            # confMat
            show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH - 1)
            show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH - 1)

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                               best_acc, best_epoch + 1))

    with open(os.path.join(log_dir, 'log.log'), 'a') as f:
        f.write(" done ~~~~ {}, best acc: {} in :{} epochs.\n ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                                       best_acc, best_epoch + 1))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
