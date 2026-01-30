import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from scipy.signal import resample
from sklearn.model_selection import train_test_split
import os
from scipy.signal import spectrogram


x_ecg = []
x_gsr = []
x_inf_ppg = []
x_pix_ppg = []
y = []

for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        
        # 100 Hz for 30 sec ->  3_000
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ecg.csv").to_numpy().transpose():
            for k in range(len(trial) // 3_000 - 1):
                x_ecg.append(list(trial[k*3_000:(k+1)*3_000].astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_gsr.csv").to_numpy().transpose():
            for k in range(len(trial) // 3_000 - 1):
                x_gsr.append(list(trial[k*3_000:(k+1)*3_000].astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ppg.csv").to_numpy().transpose():
            for k in range(len(trial) // 3_000 - 1):
                x_inf_ppg.append(list(trial[k*3_000:(k+1)*3_000].astype(np.float32)))
            
        # 256 Hz for 30 sec -> 7_680
        for trial in  pd.read_csv(f"{folder_name[0]}/pixart.csv").to_numpy().transpose():
            for k in range(len(trial) // 7_680 - 1):
                x_pix_ppg.append(list(trial[k*7_680:(k+1)*7_680].astype(np.float32)))
        for trial in pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[7, 1:7].to_numpy():
            for k in range(24): # duplicate results for the same trial (since we split the in 30s slices)
                y.append(np.float32(trial))

resample_size = 120
x_ecg_res = [resample(x, resample_size) for x in x_ecg]
x_gsr_res = [resample(x, resample_size) for x in x_gsr]
x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]
x_pix_ppg_res = [resample(x, resample_size) for x in x_pix_ppg]

print(len(x_ecg_res), len(x_gsr_res), len(x_inf_ppg_res), len(x_pix_ppg_res))

#pour x_ecg_res : 
for i in range (len(x_ecg_res)) : 
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_ecg_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_ecg_res[i] = Sxx

#pour x_gsr_res : 
for i in range (len(x_gsr_res)) : 
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_gsr_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_gsr_res[i] = Sxx

#pour x_inf_ppg_res : 
for i in range (len(x_inf_ppg_res)) : 
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_inf_ppg_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_inf_ppg_res[i] = Sxx

#pour x_pix_ppg_res : 
for i in range (len(x_pix_ppg_res)) : 
    fs = 4
    t = np.linspace(0,30,fs*30)
    signal = x_pix_ppg_res[i]
    f,time,Sxx = spectrogram(signal,fs=fs,nperseg=32,noverlap=16)
    x_pix_ppg_res[i] = Sxx

final_signal = np.stack([x_ecg_res, x_gsr_res, x_inf_ppg_res, x_pix_ppg_res], axis=0)   

print(final_signal.shape)  

#y_tensor = torch.tensor(y).float() 
#train_size = int(0.8 * len(x_all_tensor))
#train_dataloader = torch.utils.data.DataLoader(list(zip(x_all_tensor[:train_size], y_tensor[:train_size])), batch_size=32, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(list(zip(x_all_tensor[train_size:], y_tensor[train_size:])), batch_size=32, shuffle=False)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x=self.Conv1(x)
        x=torch.nn.functional.batch_norm(x, self.Conv1.weight, self.Conv1.bias)
        x=torch.nn.functional.relu(x)
        x=self.Conv2(x)
        x=torch.nn.functional.batch_norm(x, self.Conv2.weight, self.Conv2.bias)
        x=torch.nn.functional.relu(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=4, pretrained=True):
        super().__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)


        # Encoder
        self.encoder1 = nn.Sequential(self.first_conv, resnet.bn1, resnet.relu)
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3

        # Regressor
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)   
        x2 = self.pool1(x1)
        x3 = self.encoder2(x2)  
        x4 = self.encoder3(x3)  
        
        out = self.classifier(x4)

        return out