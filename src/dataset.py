import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
#from .scaler import Scaler
import numpy as np

import warnings

warnings.filterwarnings('ignore')

class TimeDataset(Dataset):
    def __init__(self, root_path='./data/processed/', seq_len=24, pred_len=8, split='train', data_path='Demand.csv',
                 target='target', scaler=None, step=1, reduce_data=1):

        assert split in ['train', 'test', 'valid']
        type_map = {'train': 0, 'valid': 1, 'test': 2}
        self.set_type = type_map[split]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.step = step
        self.reduce_data = reduce_data
        self.target = target
        self.scaler = scaler

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def _generate_time_features(self, df_stamp):
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        df_stamp['year'] = df_stamp.date.apply(lambda row: row.year, 1)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute//10, 1)

        if len(df_stamp['minute'].unique())<2: # If the minute never changes, remove it
            df_stamp = df_stamp.drop('minute', axis=1)

        df_stamp = df_stamp.drop('date', axis=1)
        return df_stamp

    def __read_data__(self):

        df_raw = pd.read_csv(self.root_path+self.data_path)

        test_size = int(len(df_raw)*0.2)

        if self.reduce_data < 1:
            df_raw = df_raw.iloc[-int(self.reduce_data*(len(df_raw)-test_size))-test_size:]

        train_size = int((len(df_raw)-test_size)*0.8)
        valid_size = int((len(df_raw)-test_size)*0.2)
        
        if 'ETTh' in self.data_path:
            border1s = [0, int(self.reduce_data*(12*30*24)) - self.seq_len, int(self.reduce_data*(12*30*24+4*30*24)) - self.seq_len]
            border2s = [int(self.reduce_data*(12*30*24)), int(self.reduce_data*(12*30*24+4*30*24)), 12*30*24+8*30*24]
        else:
            border1s = [0, train_size, train_size+valid_size]
            border2s = [train_size, train_size+valid_size, train_size+valid_size+test_size]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        border1 = int(border1)

        if self.target != 'all':
            df_raw = df_raw[[self.target, 'date']][border1:border2]
        else:
            df_raw = df_raw.drop(['group_idx', 'time_idx'], axis=1)[border1:border2]

        df_stamp = None
        if 'date' in df_raw.columns:
            df_stamp = df_raw[['date']]
            df_stamp = self._generate_time_features(df_stamp)
            df_raw = df_raw.drop('date', axis=1)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(df_raw)
        
        if self.target != 'all':
            self.data = self.scaler.transform(df_raw.values.reshape(-1, 1))
        else:
            self.data = self.scaler.transform(df_raw.values)

        if df_stamp is not None:
            self.data_stamp = df_stamp.values
    
    def target_scale_factors(self):
        
        return np.array(self.scaler.mean_), np.array(self.scaler.scale_)

    def __getitem__(self, index):
        s_begin = index*self.step
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        target_data = self.data

        seq_x = target_data[s_begin:s_end]
        seq_y = target_data[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark[:, 1:], seq_y_mark

    def __len__(self):
        return len(self.data)//self.step - self.seq_len - self.pred_len + 1