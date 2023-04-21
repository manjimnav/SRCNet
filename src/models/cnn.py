import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import ContinuousEmbedding
from .layers import ConvBlock

class CNN(nn.Module):

    def __init__(self, emb_in=1, emb_out=1, seq_len=8, pred_len=8, emb_len=128, d_model=512, n_layers=3, kernel_size=3,  dropout=0.1):
        super(CNN, self).__init__()
        self.pred_len = pred_len

        # Encoding
        self.embedding = ContinuousEmbedding(emb_in, emb_len)
        
        self.convs = nn.ModuleList([ConvBlock(emb_len, d_model, kernel_size, dropout) for l in range(n_layers)])

        self.out_layer = nn.Conv1d(seq_len, pred_len, kernel_size=1)
        self.output_layer = nn.Linear(emb_len, emb_out)

    def compute_loss(self, true, pred, criterion):
        
        return criterion(true, pred)
    
    def forward(self, inputs):
        # input -> [batch_size, seq_len, features]
        x = self.embedding(inputs)

        x = x.transpose(1, 2)  # [ batch_size, features, seq_len]

        for conv in self.convs:
            x = conv(x) 

        x = x.transpose(1, 2)  # [ batch_size, seq_len, features]

        out = self.out_layer(x)

        return self.output_layer(out)