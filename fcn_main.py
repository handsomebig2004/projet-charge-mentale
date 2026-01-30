from scipy.signal import resample
import pandas as pd
from fcn import FCNBranch, FCNModel, train, test_model
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import rnn

NUM_PATIENTS = 22


class FcnDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

x_ecg = []
x_gsr = []
x_inf_ppg = []
x_pix_ppg = []
y = []

for folder_name in os.walk("data/MAUS/Data/Raw_data/"):
    if folder_name[0][-1] != '/':
        
        frag_length = 30
        
        # 256 Hz
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
            
        # 100 Hz
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


# resample data to 4Hz on 30 seconds
resample_size = 120

x_inf_ppg_norm = torch.nn.functional.normalize(torch.tensor(x_inf_ppg))
x_ecg_norm = torch.nn.functional.normalize(torch.tensor(x_ecg))
x_gsr_norm = torch.nn.functional.normalize(torch.tensor(x_gsr))
x_pix_ppg_norm = torch.nn.functional.normalize(torch.tensor(x_pix_ppg))

def split_data(x_list, y, train_indices, valid_indices, test_indices):
    train_list = []
    valid_list = []
    test_list = []
    for x in x_list:
        train_list.append(torch.tensor([x[i] for i in train_indices]))
        valid_list.append(torch.tensor([x[i] for i in valid_indices]))
        test_list.append(torch.tensor([x[i] for i in test_indices]))
    y_train, y_valid, y_test = [y[i] for i in train_indices], [y[i] for i in valid_indices], [y[i] for i in test_indices]
    return train_list, valid_list, test_list, y_train, y_valid, y_test

indices = list(range(NUM_PATIENTS))

train_indices, test_indices = train_test_split(indices, test_size=0.2)
train_indices, valid_indices = train_test_split(train_indices, test_size=0.2)

train_indices = [x for i in range(len(x_ecg) // NUM_PATIENTS) for x in train_indices]
valid_indices = [x for i in range(len(x_ecg) // NUM_PATIENTS) for x in valid_indices]
test_indices = [x for i in range(len(x_ecg) // NUM_PATIENTS) for x in test_indices]

x_train_list, x_valid_list, x_test_list, y_train, y_valid, y_test = split_data([x_inf_ppg, x_ecg, x_gsr, x_pix_ppg], y, train_indices, valid_indices, test_indices)

print(len(x_train_list[0]))
print(len(x_valid_list[0]))
print(len(x_test_list[0]))

train_dataset = FcnDataset(list(zip(x_train_list[0], 
                               x_train_list[1], 
                               x_train_list[2], 
                               x_train_list[3])), y_train)
valid_dataset = FcnDataset(list(zip(x_valid_list[0],
                               x_valid_list[1],
                               x_valid_list[2],
                               x_valid_list[3])), y_valid)
test_dataset = FcnDataset(list(zip(x_test_list[0],
                              x_test_list[1],
                              x_test_list[2],
                              x_test_list[3],)), y_test)

# not normalized data loaders
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=12)
valid_data_loader = torch.utils.data.DataLoader(dataset=valid_dataset, shuffle=True, batch_size=12)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=True, batch_size=12)


#grouping inputs and input lengths
input_lengths = [len(x) for x in x_train_list]

fcn_net=FCNModel(num_signals=3, kernel_size=7, input_lengths=input_lengths)
rnn_net = rnn.MultiSignalRNN(num_signals=3)

loss_func = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
optim_adam = torch.optim.Adam(params= fcn_net.parameters())

#train(fcn_net, [x_ecg_train_norm_loader, x_gsr_train_norm_loader, x_inf_ppg_train_norm_loader],  [x_ecg_test_norm_loader, x_gsr_test_norm_loader, x_inf_ppg_test_norm_loader],  y_train_loader, y_test_loader, loss_func, optim_adam, n_epochs=20)
train_loss_list, valid_loss_list=train(fcn_net, train_data_loader, valid_data_loader, loss_func, optim_adam, n_epochs=15)
plt.plot(range(len(train_loss_list)), train_loss_list, label='train')

print(f'test loss (mse): {test_model(fcn_net, test_data_loader, loss_func=loss_func)}')
print(f'test loss (mae): {test_model(fcn_net, test_data_loader, loss_func=mae_loss)}')

plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid')
plt.legend()
plt.show()

