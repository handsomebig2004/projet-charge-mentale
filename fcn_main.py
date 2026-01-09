from scipy.signal import resample
import pandas as pd
from fcn import FCNBranch, FCNModel, train
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

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

print(len(x_ecg))

# resample data to 4Hz on 30 seconds
resample_size = 120
#x_ecg_res = [resample(x, resample_size) for x in x_ecg]
#x_gsr_res = [resample(x, resample_size) for x in x_gsr]
#x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]
#x_pix_ppg_res = resample(x_pix_ppg, resample_size) TODO make this work (not all things of the same size)

#x_ecg_res_norm = torch.nn.functional.normalize(torch.tensor(x_ecg_res))
#x_gsr_res_norm = torch.nn.functional.normalize(torch.tensor(x_gsr_res))
x_inf_ppg_norm = torch.nn.functional.normalize(torch.tensor(x_inf_ppg))
x_ecg_norm = torch.nn.functional.normalize(torch.tensor(x_ecg))
x_gsr_norm = torch.nn.functional.normalize(torch.tensor(x_gsr))
x_inf_ppg_norm = torch.nn.functional.normalize(torch.tensor(x_inf_ppg))

(
#x_ecg_res_norm_train,
#x_ecg_res_norm_test,
#x_gsr_res_norm_train,
#x_gsr_res_norm_test,
#x_inf_ppg_res_norm_train,
#x_inf_ppg_res_norm_test,
x_ecg_norm_train,
x_ecg_norm_test,
x_gsr_norm_train,
x_gsr_norm_test,
x_inf_ppg_norm_train,
x_inf_ppg_norm_test,
#x_ecg_res_train,
#x_ecg_res_test,
#x_gsr_res_train,
#x_gsr_res_test,
#x_inf_ppg_res_train,
#x_inf_ppg_res_test,
x_ecg_train,
x_ecg_test,
x_gsr_train,
x_gsr_test,
x_inf_ppg_train,
x_inf_ppg_test,
y_train,
y_test) = train_test_split(
    #x_ecg_res_norm,
    #x_gsr_res_norm,
    #x_inf_ppg_res_norm,
    x_ecg_norm,
    x_gsr_norm,
    x_inf_ppg_norm,
    #x_ecg_res,
    #x_gsr_res,
    #x_inf_ppg_res,
    x_ecg, 
    x_gsr, 
    x_inf_ppg,
    y,
    train_size=0.8,
    random_state=42
)

# resampled data loaders
#x_ecg_res_train_loader = torch.utils.data.DataLoader(torch.tensor(x_ecg_res_train), shuffle=True, batch_size=12)
#x_ecg_res_test_loader = torch.utils.data.DataLoader(torch.tensor(x_ecg_res_test), shuffle=True, batch_size=12)
#x_gsr_res_train_loader = torch.utils.data.DataLoader(torch.tensor(x_gsr_res_train), shuffle=True, batch_size=12)
#x_gsr_res_test_loader = torch.utils.data.DataLoader(torch.tensor(x_gsr_res_test), shuffle=True, batch_size=12)
#x_inf_res_ppg_train_loader = torch.utils.data.DataLoader(torch.tensor(x_inf_ppg_res_train), shuffle=True, batch_size=12)
#x_inf_res_ppg_test_loader = torch.utils.data.DataLoader(torch.tensor(x_inf_ppg_res_test), shuffle=True, batch_size=12)

# not resampled data loaders
x_ecg_train_loader = torch.utils.data.DataLoader(torch.tensor(x_ecg_train), shuffle=True, batch_size=12)
x_ecg_test_loader = torch.utils.data.DataLoader(torch.tensor(x_ecg_test), shuffle=True, batch_size=12)
x_gsr_train_loader = torch.utils.data.DataLoader(torch.tensor(x_gsr_train), shuffle=True, batch_size=12)
x_gsr_test_loader = torch.utils.data.DataLoader(torch.tensor(x_gsr_test), shuffle=True, batch_size=12)
x_inf_ppg_train_loader = torch.utils.data.DataLoader(torch.tensor(x_inf_ppg_train), shuffle=True, batch_size=12)
x_inf_ppg_test_loader = torch.utils.data.DataLoader(torch.tensor(x_inf_ppg_test), shuffle=True, batch_size=12)

# resampled normalized data loaders
#x_ecg_res_train_loader = torch.utils.data.DataLoader(x_ecg_res_norm_train, shuffle=True, batch_size=12)
#x_ecg_res_test_loader = torch.utils.data.DataLoader(x_ecg_res_norm_test, shuffle=True, batch_size=12)
#x_gsr_res_train_loader = torch.utils.data.DataLoader((x_gsr_res_norm_train), shuffle=True, batch_size=12)
#x_gsr_res_test_loader = torch.utils.data.DataLoader((x_gsr_res_norm_test), shuffle=True, batch_size=12)
#x_inf_res_ppg_train_loader = torch.utils.data.DataLoader((x_inf_ppg_res_norm_train), shuffle=True, batch_size=12)
#x_inf_res_ppg_test_loader = torch.utils.data.DataLoader((x_inf_ppg_res_norm_test), shuffle=True, batch_size=12)

# not resampled normalized data loaders
x_ecg_train_norm_loader = torch.utils.data.DataLoader((x_ecg_norm_train), shuffle=True, batch_size=12)
x_ecg_test_norm_loader = torch.utils.data.DataLoader((x_ecg_norm_test), shuffle=True, batch_size=12)
x_gsr_train_norm_loader = torch.utils.data.DataLoader((x_gsr_norm_train), shuffle=True, batch_size=12)
x_gsr_test_norm_loader = torch.utils.data.DataLoader((x_gsr_norm_test), shuffle=True, batch_size=12)
x_inf_ppg_train_norm_loader = torch.utils.data.DataLoader((x_inf_ppg_norm_train), shuffle=True, batch_size=12)
x_inf_ppg_test_norm_loader = torch.utils.data.DataLoader((x_inf_ppg_norm_test), shuffle=True, batch_size=12)

# y data loaders
y_train_loader = torch.utils.data.DataLoader(torch.tensor(y_train), shuffle=True, batch_size=12)
y_test_loader = torch.utils.data.DataLoader(torch.tensor(y_test), shuffle=True, batch_size=12)


#grouping inputs and input lengths
inputs= [x_ecg_train_loader, x_gsr_train_loader, x_inf_ppg_train_loader]
input_lengths = [
    max(len(s) for s in x_ecg_train),
    max(len(s) for s in x_gsr_train),
    max(len(s) for s in x_inf_ppg_train),
]

fcn_net=FCNModel(num_signals=3, kernel_size=7, input_lengths=input_lengths)


loss_func = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
optim_adam = torch.optim.Adam(params= fcn_net.parameters())

#train(fcn_net, [x_ecg_train_norm_loader, x_gsr_train_norm_loader, x_inf_ppg_train_norm_loader],  [x_ecg_test_norm_loader, x_gsr_test_norm_loader, x_inf_ppg_test_norm_loader],  y_train_loader, y_test_loader, loss_func, optim_adam, n_epochs=20)
train(fcn_net, [x_ecg_train_loader, x_gsr_train_loader, x_inf_ppg_train_loader],  [x_ecg_test_loader, x_gsr_test_loader, x_inf_ppg_test_loader],  y_train_loader, y_test_loader, loss_func, optim_adam, n_epochs=20)

