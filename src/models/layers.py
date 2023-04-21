import torch
import torch.nn as nn

class GroupLayer(nn.Module):

    def __init__(self, group_factor, group_operator='avg'):
        super(GroupLayer, self).__init__()

        self.group_factor = group_factor
        self.group_operator = group_operator
        assert group_operator in ['avg', 'sum']

    def forward(self, inputs, dim=0):
        # Inputs -> [seq_len, batch_size, features]

        inputs = inputs.unfold(dim, self.group_factor, self.group_factor)

        if self.group_operator == 'avg':
            inputs = inputs.mean(-1)
        elif self.group_operator == 'sum':
            inputs = inputs.sum(-1)

        return inputs


class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        pad = (kernel_size - 1) * dilation
        device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, padding_mode='replicate', device=device, **kwargs)
   
    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.conv.padding[0]]  # remove trailing padding
        return x 

class ConvBlock(nn.Module):

    def __init__(self,emb_len, d_model=512, kernel_size=3,  dropout=0.1, stride=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels=emb_len, out_channels=d_model, kernel_size=kernel_size)
        self.activation1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = CausalConv1d(in_channels=d_model, out_channels=emb_len, kernel_size=kernel_size)
        self.activation2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print
        x = self.conv1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.activation2(x)

        x = self.dropout(x)

        return x

class ConvBlock2(nn.Module):

    def __init__(self,emb_len, d_model=512, kernel_size=3,  dropout=0.1, stride=1, dilation=1):
        super(ConvBlock2, self).__init__()
        self.conv1 = CausalConv1d(in_channels=emb_len, out_channels=d_model, kernel_size=kernel_size,  stride=stride)
        self.activation1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)

        x = self.dropout(x)

        return x


class FCBlock(nn.Module):

    def __init__(self, emb_len, d_model=512,  dropout=0.1):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features=emb_len, out_features=d_model)
        #self.activation1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        #x = self.activation1(x)

        x = self.dropout(x)

        return x

class LSTMBlock(nn.Module):

    def __init__(self, emb_len, d_model=512,  dropout=0.1):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size=emb_len, hidden_size=d_model,batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        if type(x) == tuple and x[0].shape[2] == self.d_model:
            x = self.lstm(x[0], x[1])
        elif type(x) == tuple:
            x = self.lstm(x[0])
        else:
             x = self.lstm(x)

        return x


if __name__=='__main__':
    gl = GroupLayer(6)
    print(gl(torch.zeros((24, 32, 128))).shape)
