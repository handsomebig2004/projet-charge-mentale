from scipy.signal import resample
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
import torch


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


class LSTM(nn.Module):

    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, 384, batch_first=True)
        self.swish = nn.SiLU()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(384, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        
        x = self.conv(x)          # (batch, 64, 120)
        x = self.relu(x)

        x = x.permute(0, 2, 1)    # (batch, 120, 64)
        x, _ = self.lstm(x)       # (batch, 120, 384)

        x = x[:, -1, :]           # (batch, 384) -> dernier timestep

        x = self.swish(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.linear1(x)       # (batch, 1)


        return x

    
resample_size = 120
x_ecg_res = [resample(x, resample_size) for x in x_ecg]
x_gsr_res = [resample(x, resample_size) for x in x_gsr]
x_inf_ppg_res = [resample(x, resample_size) for x in x_inf_ppg]
#x_pix_ppg_res = resample(x_pix_ppg, resample_size) 

#concaténer mes signaux
x_all = []
for i in range(len(x_ecg_res)):
    signals = np.stack([x_ecg_res[i], x_gsr_res[i], x_inf_ppg_res[i]], axis=0)
    x_all.append(signals)


x_all_tensor = torch.tensor(np.array(x_all)).float()
y_tensor = torch.tensor(y).float() 

train_size = int(0.8 * len(x_all_tensor))
train_dataloader = torch.utils.data.DataLoader(list(zip(x_all_tensor[:train_size], y_tensor[:train_size])), batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(list(zip(x_all_tensor[train_size:], y_tensor[train_size:])), batch_size=32, shuffle=False)

n_epochs = 20
model = LSTM(input_size=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def valid_epoch(x_batch, y_batch, loss_func, model):
    preds = model(x_batch)
    loss = loss_func(preds, y_batch)
    return loss

for epoch in range(n_epochs):
    for x_batch, y_batch in train_dataloader:
        model.train()
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        valid_loss = valid_epoch(test_dataloader, loss_fn, model)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, Valid loss; {valid_loss:.4f}")