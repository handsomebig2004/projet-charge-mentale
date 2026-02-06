import torch
import torch.nn as nn
from torchvision import models
from load_data import train_freq_data_loader, valid_freq_data_loader, test_freq_data_loader
import matplotlib.pyplot as plt

models_list = []
optimizer_list = []
loss_list = []

def feature_extraction(y_batch,i) :
    res = []
    for sample in y_batch : 
        res.append(sample[i])
    return res

for i in range(6) : 
     
    model = models.resnet18(weights='DEFAULT')
    model.conv1 = nn.Conv2d(4,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.fc = torch.nn.Linear(in_features=512, out_features=1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    models_list.append(model)    
    optimizer_list.append(optimizer)
    loss_list.append(criterion)

n_epochs = 50

train_loss_list = []
valid_loss_list = []

def valid_epoch(test_loader, loss_func, model):
    
    model.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:

            preds = model(x_batch)

            loss = loss_func(preds.squeeze(), y_batch)
            
            n_samples += y_batch.size(0)
            tot_loss += loss.item() * y_batch.size(0)

    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    valid_loss_list.append(avg_loss)
    return avg_loss

epoch_loss = 0
n_samples = 0

for epoch in range(n_epochs):
    for x_batch, y_batch in train_freq_data_loader:
        
        for i in range(6) :
            y = feature_extraction(y_batch,i)
            y = torch.tensor(y, dtype=torch.float32)
            models_list[i].train()
            optimizer_list[i].zero_grad()
            outputs = models_list[i](x_batch)
            loss = loss_list[i](outputs.squeeze(), y)
            loss.backward()
            optimizer_list[i].step()
            epoch_loss += loss.item() * y.size(0)
            n_samples += y.size(0)

