from scipy.signal import resample
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from rnn import *

x_ecg = []
x_gsr = []
x_inf_ppg = []
x_pix_ppg = []
y = []

for folder_name in os.walk("data/MAUS/Data/Normalized_data/"):
    if folder_name[0][-1] != '/':
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ecg.csv").to_numpy().transpose():
            x_ecg.append(list(trial))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_gsr.csv").to_numpy().transpose():
            x_gsr.append(list(trial))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ppg.csv").to_numpy().transpose():
            x_inf_ppg.append(list(trial))
        for trial in  pd.read_csv(f"{folder_name[0]}/pixart.csv").to_numpy().transpose():
            x_pix_ppg.append(list(trial))
        for trial in pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[7, 1:7].to_numpy():
            y.append(float(np.float32(trial) / 100.0))

# resample data to 4Hz on 30 seconds
resample_size = 120
x_ecg_res = [resample(x, resample_size) for x in x_ecg]
x_gsr_res = [resample(x, resample_size) for x in x_gsr]
x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]
#x_pix_ppg_res = resample(x_pix_ppg, resample_size) TODO make this work (not all things of the same size)

x_ecg_res_train, x_ecg_res_test, x_gsr_res_train, x_gsr_res_test, x_inf_ppg_res_train, x_inf_ppg_res_test, x_ecg_train, x_ecg_test, x_gsr_train, x_gsr_test, x_inf_ppg_train, x_inf_ppg_test, y_train, y_test = train_test_split(
    x_ecg_res,
    x_gsr_res,
    x_inf_ppg_res,
    x_ecg, 
    x_gsr, 
    x_inf_ppg,
    y,
    train_size=0.8
)

# resampled data loaders
x_ecg_res_train_loader = torch.utils.data.DataLoader(x_ecg_res_train, shuffle=True, batch_size=12)
x_ecg_res_test_loader = torch.utils.data.DataLoader(x_ecg_res_test, shuffle=True, batch_size=12)
x_gsr_res_train_loader = torch.utils.data.DataLoader(x_gsr_res_train, shuffle=True, batch_size=12)
x_gsr_res_test_loader = torch.utils.data.DataLoader(x_gsr_res_test, shuffle=True, batch_size=12)
x_inf_res_ppg_train_loader = torch.utils.data.DataLoader(x_inf_ppg_res_train, shuffle=True, batch_size=12)
x_inf_res_ppg_test_loader = torch.utils.data.DataLoader(x_inf_ppg_res_test, shuffle=True, batch_size=12)

# not resampled data loaders
x_ecg_train_loader = torch.utils.data.DataLoader(x_ecg_train, shuffle=True, batch_size=12)
x_ecg_test_loader = torch.utils.data.DataLoader(x_ecg_test, shuffle=True, batch_size=12)
x_gsr_train_loader = torch.utils.data.DataLoader(x_gsr_train, shuffle=True, batch_size=12)
x_gsr_test_loader = torch.utils.data.DataLoader(x_gsr_test, shuffle=True, batch_size=12)
x_inf_ppg_train_loader = torch.utils.data.DataLoader(x_inf_ppg_train, shuffle=True, batch_size=12)
x_inf_ppg_test_loader = torch.utils.data.DataLoader(x_inf_ppg_test, shuffle=True, batch_size=12)

# y data loaders
y_train_loader = torch.utils.data.DataLoader(y_train, shuffle=True, batch_size=12)
y_test_loader = torch.utils.data.DataLoader(y_test, shuffle=True, batch_size=12)


rnn_net = MultiSignalRNN(num_signals=3, hidden_size=128, num_layers=1)
loss_func = torch.nn.MSELoss()
optim_adam = torch.optim.Adam(params=rnn_net.parameters(), lr=1e-3)

train_rnn(
    rnn_net,
    [x_ecg_train_loader, x_gsr_train_loader, x_inf_ppg_train_loader],
    [x_ecg_test_loader,  x_gsr_test_loader,  x_inf_ppg_test_loader],
    y_train_loader,
    y_test_loader,
    loss_func,
    optim_adam,
    n_epochs=5
)
