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
        x = self.linear1(x)
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

def train(dataloader_train, dataloader_test, network,optimizer, loss, nepochs):
    acc_test=[]
    network.train()
    print(f'Initially: loss {loss(network, dataloader_train)}')
    for e in range(nepochs):
        epoch(dataloader_train,network,optimizer,loss)
        acc_test.append(accuracy(network, dataloader_test))
        print(f'Epoch {e+1} : {acc_test[-1]}')

    return acc_test

# tester transformers avec 1 ou 2 heads
# slide avec resultats jusque là
# chercher autre archi dans l'état de l'art pour charge mentale
