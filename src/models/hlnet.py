import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyTask(nn.Module):

    def __init__(self, emb_in, seq_len, pred_len, emb_len, d_model, n_layers, dropout, pool_size, pool_stride):
        super(FrequencyTask, self).__init__()
        self.pool = nn.AvgPool1d(pool_size, stride=pool_stride)
        self.lstm = nn.LSTM(emb_in, d_model, num_layers =n_layers, dropout=dropout)

        self.output_layer = nn.Linear(d_model, pred_len-pool_size+1)
    
    def forward(self, x):
        x = self.pool(x.transpose(1,2)).transpose(1,2)
        x, _ = self.lstm(x)

        x = x[:, -1, :]

        return self.output_layer(x).unsqueeze(-1), x
        

class HLNet(nn.Module):

    def __init__(self, emb_in=1, seq_len=8, pred_len=8, emb_len=128, d_model=512, dropout=0.1, n_layers=1, smooth_factor=6):
        super(HLNet, self).__init__()

        self.pred_len = pred_len
        self.auxiliar_features = []

        self.pools_and_strides = [(smooth_factor, 1)]

        self.auxiliars_lstm_blocks = nn.ModuleList([ FrequencyTask(emb_in, seq_len, pred_len, emb_len, d_model, n_layers, dropout, pool_size, pool_stride)
                                                    for pool_size, pool_stride in self.pools_and_strides])

        self.principal_lstm = nn.LSTM(emb_in, d_model, num_layers=n_layers, dropout=dropout)

        self.out_layer = nn.Linear(d_model*(len(self.auxiliars_lstm_blocks)+1), pred_len)

    def consistency_loss(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def compute_loss(self, true, pred, criterion):

        auxiliar_loss = 0

        for (pool, stride), output in zip(self.pools_and_strides, self.auxiliar_outputs):
            pool_function = nn.AvgPool1d(pool, stride=stride)
            true_pooled = pool_function(true.transpose(1, 2)).transpose(1, 2)
            auxiliar_loss +=criterion(true_pooled, output)

        principal_loss = criterion(true, pred)

        return principal_loss, auxiliar_loss
    
    def forward(self, inputs):
        # input -> [batch_size, seq_len, features]
        x = inputs

        self.auxiliar_outputs = []
        self.auxiliar_features = []
        for auxiliar_task in self.auxiliars_lstm_blocks:
            output, features = auxiliar_task(x)
            self.auxiliar_features.append(features.detach().clone())
            self.auxiliar_outputs.append(output)
        
        x_auxiliar = torch.cat(self.auxiliar_features, -1)

        x, _ = self.principal_lstm(x)
        x = x[:, -1, :]

        x = torch.cat((x_auxiliar, x), -1)

        return self.out_layer(x).unsqueeze(-1)

