import torch
import torch.nn.functional as F
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, emb_in=1, emb_out=1, seq_len=8, pred_len=8, d_model=512, n_layers=2, dropout=0.1):
        super(LSTM, self).__init__()

        self.pred_len = pred_len
        self.emb_in = emb_in
        self.emb_out = emb_out

        self.lstm = nn.LSTM(input_size=emb_in, hidden_size=d_model, batch_first=True, num_layers = n_layers, dropout=dropout)
    
        self.output = nn.Linear(seq_len*d_model, pred_len*emb_out)

    def compute_loss(self, true, pred, criterion):

        return criterion(true, pred) 

    def forward(self, inputs):

        x, _ = self.lstm(inputs)

        return self.output(x.flatten(1)).reshape(-1, self.pred_len, self.emb_out)