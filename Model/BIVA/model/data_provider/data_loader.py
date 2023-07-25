import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler, load_data_DFM, set_lag_DFM, repeat_row
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_BIVA(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='custom.csv',
                 target='GDP', scale=True, inverse=False, timeenc=0, freq='m', cols=None):
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.start_M = '2000-01'
        self.end_M = '2022-12'
        self.start_Q = '1999-03'
        self.end_Q = '2023-03'
        
        self.load_data_DFM = load_data_DFM
        self.set_lag_DFM = set_lag_DFM
        self.repeat_row = repeat_row
        
        self.__read_data__()


    def __read_data__(self):
        self.scaler_m = StandardScaler()
        self.scaler_q = StandardScaler()
        
        path = os.path.join(self.root_path, self.data_path)
        
        df_Q, df_Q_trans, df_M, df_M_trans, self.var_info = self.load_data_DFM(path)
        
        cols_M = list(df_M.columns)
        cols_Q = list(df_Q.columns)
        cols_Q.remove(self.target)
        df_Q = df_Q[cols_Q + [self.target]]
        df_M = df_M.loc[self.data_start_M, self.data_end_M, cols_M]
        df_Q = df_M.loc[self.data_start_Q, self.data_end_Q, cols_Q].repeat(3)
        print(f"df_M.shape period (start: {self.start_M} ~ end: {self.end_M}): {df_M.shape}")
        print(f"df_Q.shape period (start: {self.start_Q} ~ end: {self.end_Q}): {df_Q.shape}")
        # print(f"df_raw.cols : {df_raw.columns}")

        num_train = int(len(df_M) * (0.8 if not self.train_only else 1))
        num_test = int(len(df_M) * 0.1)
        num_vali = len(df_M) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_M) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_M)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
        elif self.set_type == 1:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
        else:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            train_data_t = df_data_t[border1s[0]:border2s[0]]
            self.scaler_m.fit(train_data.values)
            data = self.scaler_m.transform(df_data.values)
            self.scaler_q.fit(train_data_t.values)
            data_t = self.scaler_q.transform(df_data_t.values)
        else:
            data = df_data.values
            data_t = df_data_t.values

        # data_stamp_data_M
        df_stamp = df_M.index  # [border1:border2]
        # df_stamp = pd.to_datetime(df_stamp)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.values
        elif self.timeenc == 1:
            # data_stamp = time_features(df_stamp.values, freq=self.freq)
            # data_stamp = data_stamp.transpose(1, 0)
            data_stamp = df_stamp.values

        # data_stamp_target_Q
        df_stamp_t = df_Q.index
        # df_stamp_t = df_Q[['date']]  # [border1:border2]
        # df_stamp_t['date'] = pd.to_datetime(df_stamp_t.date)
        if self.timeenc == 0:
            df_stamp_t['month'] = df_stamp_t.apply(lambda row: row.month, 1)
            # df_stamp_t['month'] = df_stamp_t.date.apply(lambda row: row.month, 1)
            df_stamp_t['day'] = df_stamp_t.apply(lambda row: row.day, 1)
            df_stamp_t['weekday'] = df_stamp_t.apply(lambda row: row.weekday(), 1)
            df_stamp_t['hour'] = df_stamp_t.apply(lambda row: row.hour, 1)
            data_stamp_t = df_stamp_t.drop(['date'], 1).values
        elif self.timeenc == 1:
            # data_stamp_t = time_features(pd.to_datetime(df_stamp_t['date'].values), freq=self.freq)
            # data_stamp_t = data_stamp_t.transpose(1, 0)
            data_stamp_t = df_stamp_t.values

        self.data_x = data[border1:border2]
        self.data_y = data_t[border1:border2]
        self.data_stamp = data_stamp
        self.data_stamp_t = data_stamp_t

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # state period is Q freq (so, take one more time step to forward)
        r_begin = s_end - self.label_len - 1
        r_end = r_begin + self.lable_len + self.pred_len - 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # set lag seq_x
        seq_x = self.set_lag_DFM(seq_x, self.var_info,'M')
        seq_x_mark = self.set_lag_DFM(seq_x_mark, self.var_info,'M')

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_DFM(Dataset):
    pass    
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, features='MS', data_path='custom.csv',
                 target='GDP', scale=True, inverse=False, timeenc=0, freq='m', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_M = pd.read_excel(os.path.join(self.root_path, self.data_path),
                             index_col='date', sheet_name='df_M', header=0)
        p_rng_m = pd.period_range('1970-01-01', '2023-06-01', freq='m')
        df_M = df_M.set_index(p_rng_m)
        df_M = df_M.iloc[:, :].astype('float')
        df_M.index.name = 'date'

        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)

        cols.remove('mea_dt')
        df_raw.rename(columns={'mea_dt': 'date'}, inplace=True)
        # target value position define
        df_raw = df_raw[['date'] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
