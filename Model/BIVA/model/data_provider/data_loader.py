import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
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
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_M = pd.read_excel(os.path.join(self.root_path, self.data_path),
                             index_col='date', sheet_name='df_M', header=0)
        p_rng_m = pd.period_range('1970-01-01', '2023-06-01', freq='m')
        df_M = df_M.set_index(p_rng_m)
        df_M = df_M.iloc[:, :].astype('float')
        df_M.index.name = 'date'

        # Quater
        df_Q = pd.read_excel(os.path.join(self.root_path, self.data_path),
                             index_col='date', sheet_name='df_Q', header=0)
        p_rng_q = pd.period_range('1960Q2', '2023Q1', freq='Q-FEB')
        df_Q = df_Q.set_index(p_rng_q)
        df_Q = df_Q.iloc[:, :].astype('float')
        df_Q.index.name = 'date'

        # print(f"df_raw.cols : {df_raw.columns}")
        cols = list(df_M)
        # cols.remove(self.target)
        # cols.remove('mea_dt')

        # df_raw.rename(columns={'mea_dt': 'date'}, inplace=True)
        # target value position define
        # df_raw = df_M[['date'] + cols + [self.target]]

        if self.set_type == 0:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols]
                df_data_t = df_Q[[self.target]]
        elif self.set_type == 1:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols]
                df_data_t = df_Q[[self.target]]
        else:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols]
                df_data_t = df_Q[[self.target]]

        if self.scale:
            train_data = df_data
            train_data_t = df_data_t
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            self.scaler.fit(train_data_t.values)
            data_t = self.scaler.transform(df_data_t.values)
        else:
            data = df_data.values
            data_t = df_data_t.values

        # data_stamp_data_M
        df_stamp = df_M[['date']]  # [border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(
                df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # data_stamp_target_Q
        df_stamp_t = df_Q[['date']]  # [border1:border2]
        df_stamp_t['date'] = pd.to_datetime(df_stamp_t.date)
        if self.timeenc == 0:
            df_stamp_t['month'] = df_stamp_t.date.apply(
                lambda row: row.month, 1)
            df_stamp_t['day'] = df_stamp_t.date.apply(lambda row: row.day, 1)
            df_stamp_t['weekday'] = df_stamp_t.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp_t['hour'] = df_stamp_t.date.apply(lambda row: row.hour, 1)
            data_stamp_t = df_stamp_t.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp_t = time_features(pd.to_datetime(
                df_stamp_t['date'].values), freq=self.freq)
            data_stamp_t = data_stamp_t.transpose(1, 0)

        self.data_x = data  # [border1:border2]
        self.data_y = data_t  # [border1:border2]
        self.data_stamp = data_stamp
        self.data_stamp_t = data_stamp_t

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        r_begin = index
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


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