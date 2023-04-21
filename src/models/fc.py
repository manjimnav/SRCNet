import torch
import torch.nn.functional as F
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, batch_size=32, emb_in=1, emb_out=1, seq_len=8, pred_len=8, d_model=512, n_layers=2):
        super(FullyConnected, self).__init__()

        self.pred_len = pred_len
        self.emb_in = emb_in
        self.emb_out = emb_out

        self.layers = nn.ModuleList([nn.Linear(emb_in*seq_len if i == 0 else d_model, d_model) for i in range(n_layers)])
    
        self.output = nn.Linear(d_model, pred_len*emb_out)

    def compute_loss(self, true, pred, criterion):

        return criterion(true, pred) 

    def forward(self, inputs):

        x = inputs.flatten(1)
        for layer in self.layers:
            x = layer(x)

        return self.output(x).reshape(-1, self.pred_len, self.emb_out)