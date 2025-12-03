import torch
import torch.nn as nn
import torch.optim as optim

class LSTM(nn.Module):

    def __init__(self, input_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(input_size, 64)
        self.lstm = nn.LSTM(64, 384, batch_first=True)
        self.swish = nn.SiLU()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(384, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x, _ = self.lstm(x)
        x = self.swish(x)
        x = self.relu(x)
        x = self.self.linear1(x)
        x = self.dropout(x)
        return x
    


def epoch(dataloader,network,optimizer,loss) :
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = network(images)
        loss_value = loss(outputs, labels)
        loss_value.backward()
        #print(loss_value.item())
        optimizer.step()


def accuracy(network,dataloader) :
    network.eval()
    with torch.no_grad():
        acc_list = []
        for images, labels in dataloader:
            outputs = network(images)
            acc_list.append((torch.argmax(outputs, dim = 1) == labels).sum()/32)
        return(sum(acc_list)/len(acc_list))


def train(dataloader_train,dataloader_test,network,optimizer,loss, nepochs):
    acc_test=[]
    network.train()
    print(f'Initially: accuracy {accuracy(network, dataloader_train)}')
    for e in range(nepochs):
        epoch(dataloader_train,network,optimizer,loss)
        acc_test.append(accuracy(network, dataloader_test))
        print(f'Epoch {e+1} : {acc_test[-1]}')

    return acc_test