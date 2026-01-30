import os
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from scipy.signal import resample
from scipy.signal import spectrogram

x_ecg = []
x_gsr = []
x_inf_ppg = []
x_pix_ppg = []
y = []

for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        
        frag_length = 30
        
        # 256 Hz for 30 sec -> 7_680
        n_input_256 = 256 * frag_length
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ecg.csv").to_numpy().transpose():
            for k in range(len(trial) // n_input_256 - 1):
                x_ecg.append(list(trial[k*n_input_256:(k+1)*n_input_256].astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_gsr.csv").to_numpy().transpose():
            for k in range(len(trial) // n_input_256 - 1):
                x_gsr.append(list(trial[k*n_input_256:(k+1)*n_input_256].astype(np.float32)))
        for trial in pd.read_csv(f"{folder_name[0]}/inf_ppg.csv").to_numpy().transpose():
            for k in range(len(trial) // n_input_256 - 1):
                x_inf_ppg.append(list(trial[k*n_input_256:(k+1)*n_input_256].astype(np.float32)))
            
        # 100 Hz for 30 sec ->  3_000
        n_input_100 = 100 * frag_length
        for trial in  pd.read_csv(f"{folder_name[0]}/pixart.csv").to_numpy().transpose():
            #print(trial.shape)
            for k in range(len(trial) // n_input_100 - 1):
                x_pix_ppg.append(list(trial[k*n_input_100:(k+1)*n_input_100].astype(np.float32)))

for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        for trial in pd.read_csv(f"data/MAUS/Subjective_rating/{folder_name[0][-3:]}/NASA_TLX.csv").iloc[0:6, 1:7].to_numpy().transpose():
            for k in range(int(len(x_ecg) / 132)): # duplicate results for the same trial (since we split the in 30s slices)
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
final_signal = final_signal.transpose(1,0,2,3) 

print(final_signal.shape)  

y_tensor = torch.tensor(y).float() 
train_size = int(0.8 * len(final_signal))

train_dataloader = torch.utils.data.DataLoader(list(zip(final_signal[:train_size], y_tensor[:train_size])), batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(list(zip(final_signal[train_size:], y_tensor[train_size:])), batch_size=32, shuffle=False)


model = models.resnet18(weights='DEFAULT')
model.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
model.fc = torch.nn.Linear(in_features=2048, out_features=1)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
	param.requires_grad = True

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 50

def valid_epoch(test_loader, loss_func, model):
    
    model.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batche_l, y_batch in test_loader:
            #preparing all inputs

            preds = model(x_batche_l)
            
            if tot_loss == 0:
                pass
                #print(preds[:5])
                #print(y[:5])
                # plt.plot(range(len(preds)), preds)
                # plt.plot(range(len(y)), y)
                # plt.show()

            loss = loss_func(preds.squeeze(), y_batch)
            
            n_samples += y_batch.size(0)
            tot_loss += loss.item() * y_batch.size(0)

    model.train()
    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    return avg_loss

for epoch in range(n_epochs):
    for x_batch, y_batch in train_dataloader:
        model.train()
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        valid_loss = valid_epoch(test_dataloader, criterion, model)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Valid loss; {valid_loss:.4f}")