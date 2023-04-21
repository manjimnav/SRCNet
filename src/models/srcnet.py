import torch
import torch.nn.functional as F
import torch.nn as nn

from .embedding import ContinuousEmbedding
from .layers import ConvBlock


class SmoothResidualBlock(nn.Module):

    def __init__(self, seq_len, pred_len, emb_in, emb_len, d_model, kernel_size, dropout, pool_size, pool_stride, previous_seq_len=0):
        super(SmoothResidualBlock, self).__init__()
        self.pool = nn.AvgPool1d(pool_size, stride=pool_stride)
        self.conv_block = ConvBlock(emb_len, d_model, kernel_size, dropout)
        self.seq_len = seq_len
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        self.output_conv = nn.Conv1d(seq_len, pred_len, kernel_size=1)
        self.output_layer = nn.Linear(emb_len, emb_in)
    
    def forward(self, x):
        if self.pool_stride==1:
            x = F.pad(x, (self.pool_size-1, 0), mode='replicate')
        else:
            x = F.pad(x, (self.seq_len//self.pool_stride, 0), mode='replicate')

        x_pooled = self.pool(x)
        x = self.conv_block(x_pooled)
        x_conv = self.output_conv(x.transpose(1, 2))

        return self.output_layer(x_conv), x_pooled
        
        

class SRCNet(nn.Module):

    def __init__(self, emb_in=1, seq_len=8, pred_len=8, emb_len=128, d_model=512, n_layers=3, smooth_factor=2, kernel_size=3,  dropout=0.1, use_frequency_consistency=False):
        super(SRCNet, self).__init__()
        self.pred_len = pred_len

        self.use_frequency_consistency = use_frequency_consistency
        self.auxiliar_outputs = []
        self.auxiliar_features = []
        self.principal_features = None
        self.emb_in = emb_in
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Encoding
        self.embedding = ContinuousEmbedding(emb_in, emb_len)

        self.smooth_residual_blocks = nn.ModuleList([ ])
                                                    
        for i in range(n_layers+1):

            smooth_factor_layer = smooth_factor
            if i == n_layers:
                smooth_factor_layer = 1

            smooth_residual_block = SmoothResidualBlock(seq_len, pred_len, emb_in,  emb_len, d_model, kernel_size, dropout, smooth_factor_layer, 1, previous_seq_len=0)

            self.smooth_residual_blocks.append(smooth_residual_block)


    def compute_loss(self, true, pred, criterion):
        principal_loss = criterion(true, pred)

        return principal_loss

    
    def forward(self, inputs):
        # input -> [batch_size, seq_len, features]
        x = self.embedding(inputs)

        x = x.transpose(1, 2)  # [ batch_size, features, seq_len]
        self.auxiliar_features = []
        predicted_outputs = torch.zeros(x.shape[0], self.pred_len, self.emb_in, dtype=torch.float64, device=self.device)
        for i, smooth_residual_block in enumerate(self.smooth_residual_blocks):
            output_smooth, input_smooth = smooth_residual_block(x)
            predicted_outputs += output_smooth

            x = x-input_smooth

        predicted_outputs = predicted_outputs  # [ batch_size, seq_len, features]

        return predicted_outputs

