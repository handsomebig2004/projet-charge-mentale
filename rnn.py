import torch
import torch.nn as nn

def _prepare(y):
    if isinstance(y, (list, tuple)):
        y = y[0]
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    if y.dim() == 0:
        y = y.unsqueeze(0)
    return y.float()


class MultiSignalRNN(nn.Module):
    def __init__(self, num_signals, hidden_size=128, num_layers=1, dropout=0.0):
        """
        num_signals: how many signals
        hidden_size: RNN hidden layer size
        num_layers: how many layers
        """
        super().__init__()

        self.num_signals = num_signals
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        input_size = num_signals  

        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_list):
        """
        x_list: list,length num_signals,shape: [batch, L]
        """

        x_list = [_prepare(x) for x in x_list]
        
        feats = [p.unsqueeze(-1) for p in x_list] # [batch, L, 1]
        x = torch.cat(feats, dim=-1)  # [batch, L, num_signals]

        if x.dim() == 2:
            x = x.unsqueeze(0)  # -> [1, L, num_signals]

        output, hidden = self.rnn(x)

        last_output = output[:, -1, :]  # [batch, hidden_size]

        y_hat = self.fc(last_output)    # [batch, 1]
        return y_hat



def epoch_train_rnn(net, train_loader_l, y_train_loader, loss_func, optim):
    net.train()
    tot_loss, n_samples = 0.0, 0

    for *x_batches, y_batch in zip(*train_loader_l, y_train_loader):
        optim.zero_grad()

        y = _prepare(y_batch)

        preds = net(x_batches).squeeze()   # [batch] or [batch, 1] -> [batch]

        loss = loss_func(preds, y.float())
        loss.backward()
        optim.step()

        batch_size = y.size(0)
        n_samples += batch_size
        tot_loss += loss.item() * batch_size

    return tot_loss / n_samples


def epoch_valid_rnn(net, valid_loader_l, y_valid_loader, loss_func):
    net.eval()
    tot_loss, n_samples = 0.0, 0


    with torch.no_grad():
        for *x_batches, y_batch in zip(*valid_loader_l, y_valid_loader):
            y = _prepare(y_batch)
            preds = net(x_batches).squeeze()
            loss = loss_func(preds, y.float())

            batch_size = y.size(0)
            n_samples += batch_size
            tot_loss += loss.item() * batch_size

    return tot_loss / n_samples if n_samples > 0 else 0.0


def train_rnn(net, train_loader_l, valid_loader_l, y_train_loader, y_valid_loader,
              loss_func, optim, n_epochs):
    train_loss_list, valid_loss_list = [], []

    for epoch in range(n_epochs):
        train_loss = epoch_train_rnn(net, train_loader_l, y_train_loader, loss_func, optim)
        valid_loss = epoch_valid_rnn(net, valid_loader_l, y_valid_loader, loss_func)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        print(f"[RNN] Epoch {epoch}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")

    return train_loss_list, valid_loss_list