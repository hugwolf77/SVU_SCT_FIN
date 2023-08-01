import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler, load_data_timeindex, load_data_DFM, set_lag_missing, repeat_label_row
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
        self.period = {'M': ['2000-01','2023-01'], 'Q':['2000-03','2023-03']}
        # self.period = {'M': ['2010-01','2023-01'], 'Q':['2010-03','2023-03']}
        self.start_M = self.period['M'][0]
        self.end_M = self.period['M'][1]
        self.start_Q = self.period['Q'][0]
        self.end_Q = self.period['Q'][1]
        
        self.load_data_DFM = load_data_DFM
        self.load_data_timeindex = load_data_timeindex
        self.set_lag_missing = set_lag_missing
        self.repeat_label_row = repeat_label_row
        
        self.__read_data__()


    def __read_data__(self):
        self.scaler_m = StandardScaler()
        self.scaler_q = StandardScaler()
        
        path = os.path.join(self.root_path, self.data_path)
        
        # df_Q, df_Q_trans, df_M, df_M_trans, self.var_info = self.load_data_DFM(path)
        df_Q, df_Q_trans, df_M, df_M_trans, self.var_info = self.load_data_timeindex(path)
        
        # df_Q = df_Q_trans
        # df_M = df_M_trans
        
        cols_M = list(df_M.columns)
        cols_Q = list(df_Q.columns)
        cols_Q.remove(self.target)
        df_Q = df_Q[cols_Q + [self.target]]
        df_M = df_M.loc[self.start_M:self.end_M]
        df_Q = df_Q.loc[self.start_Q:self.end_Q]
        # df_Q = repeat_label_row(df=df_Q, pred_len=self.pred_len, repeat=3)
        
        # print(f"df_M.shape period (start: {self.start_M} ~ end: {self.end_M}): {df_M.shape}")
        # print(f"df_Q.shape period (start: {self.start_Q} ~ end: {self.end_Q}): {df_Q.shape}")
        # print(f"df_raw.cols : {df_raw.columns}")

        # M
        num_train = int(len(df_M) * 0.8) #(0.8 if not self.train_only else 1))
        num_test = int(len(df_M) * 0.1)
        num_vali = len(df_M) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_M) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_M)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Q       
        # num_train_Q = int(len(df_Q) * 0.8) #(0.8 if not self.train_only else 1))
        # num_test_Q = int(len(df_Q) * 0.1)
        # num_vali_Q = len(df_Q) - num_train_Q - num_test_Q
        # border1s_Q = [0, num_train_Q - self.seq_len, len(df_Q) - num_test_Q - self.seq_len]
        # border2s_Q = [num_train_Q, num_train_Q + num_vali_Q, len(df_Q)]
        border1s_Q = [0, (num_train - self.seq_len)//3, (len(df_M) - num_test - self.seq_len)//3]
        border2s_Q = [num_train//3, (num_train + num_vali)//3, len(df_M)//3]

        border1_Q = border1s_Q[self.set_type]
        border2_Q = border2s_Q[self.set_type]

        # if self.set_type == 1 or self.set_type == 2:
        #   print(f"val == > df_Q.shape: {df_Q.shape}")
        #   print(f"border1_Q: {border1_Q}")
        #   print(f"border2_Q: {border2_Q}")
        #   raise
        # else:
        #   print(f"train == > df_Q.shape: {df_Q.shape}")
        #   print(f"border1_Q: {border1_Q}")
        #   print(f"border2_Q: {border2_Q}")     

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
            df_data_cols = df_data.columns
            df_data_index = df_data.index
            
            train_data_t = df_data_t[border1s_Q[0]:border2s_Q[0]]
            df_data_t_cols = df_data_t.columns
            df_data_t_index = df_data_t.index
            
            self.scaler_m.fit(train_data.values)
            data = self.scaler_m.transform(df_data.values)
            self.scaler_q.fit(train_data_t.values)
            data_t = self.scaler_q.transform(df_data_t.values)
            
            data = pd.DataFrame(data,columns=df_data_cols, index=df_data_index)
            data_t = pd.DataFrame(data_t,columns=df_data_t_cols, index=df_data_t_index)
            
        else:
            data = df_data #.values
            data_t = df_data_t #.values

        # data_stamp_data_M
        df_stamp =  data.index
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp_t.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            data_stamp = df_stamp.values

        # data_stamp_target_Q
        df_stamp_t = data_t.index # df_Q.index[border1:border2]
        # df_stamp_t['date'] = pd.to_datetime(df_stamp_t.date)
        if self.timeenc == 0:
            df_stamp_t['month'] = df_stamp_t.apply(lambda row: row.month, 1)
            df_stamp_t['day'] = df_stamp_t.apply(lambda row: row.day, 1)
            df_stamp_t['weekday'] = df_stamp_t.apply(lambda row: row.weekday(), 1)
            df_stamp_t['hour'] = df_stamp_t.apply(lambda row: row.hour, 1)
            data_stamp_t = df_stamp_t.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp_t = time_features(pd.to_datetime(df_stamp_t.values), freq=self.freq)
            data_stamp_t = data_stamp_t.transpose(1, 0)
            data_stamp_t = df_stamp_t.values

        self.data_x = data[border1:border2]
        self.data_y = data_t[border1_Q:border2_Q]
        self.data_stamp = data_stamp[border1:border2]
        self.data_stamp_t = data_stamp_t[border1_Q:border2_Q]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        # set lag seq_x
        seq_x = self.set_lag_missing(seq_x, self.var_info,'M').values
        
        # calculate the start position of the label 
        # considering the predict length and quarter index         
        # state period is Q freq (so, take one more time step to forward)
        r_q_index = s_end//3 # quater's month length
        r_q_res   = s_end%3
        set_pred_len = self.repeat_label_row(df=self.data_y,pred_len=self.pred_len,repeat=3) 
        if r_q_res == 0:
            r_begin =  r_q_index*self.pred_len - self.pred_len
            r_end = r_begin + self.pred_len
        else:
            r_begin =  r_q_index*self.pred_len - self.pred_len + (self.pred_len*r_q_res)
            r_end = r_begin + self.pred_len
        seq_y = set_pred_len[r_begin:r_end].values
        
        # time feagure index       
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, #seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

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
