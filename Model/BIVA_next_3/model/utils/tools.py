import os
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import torch

import matplotlib.pyplot as plt


plt.switch_backend('agg')

def repeat_label_row(df,pred_len,repeat):
  stact_block = pd.DataFrame()
  length = len(df) - pred_len +1
  for i in range(length):
    pred_block = df.iloc[i:i+pred_len,:]
    for c in range(1,repeat+1):
      stact_block = pd.concat([stact_block,pred_block])
  return stact_block

def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[1]
    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan
    return treated

def adf_test(transformed):
    if (result := adfuller(transformed.values))[1] < 0.05:
        test_result = "{}".format("S")
    else:
        test_result = "{}".format("N")
    return test_result
        
def transform_adf(df, var_info):
    df_trans = pd.DataFrame(columns=df.columns,index=df.index)
    for col in df.columns:
        # print(f"transform_column : {col}")
        # print(f"transform_diff : {diff}")
        diff = var_info[var_info['ID'] == col]['transform'].values[0]
        lag = var_info[var_info['ID'] == col]['LAG'].values[0]
        # transform N test
        if diff == 'Origin':
            transformed = df[col]
            treated = remove_outliers(transformed)
            # res = adf_test(transformed.dropna())
            df_trans[col] = transformed
            # df_trans = pd.concat([df_trans,treated],axis=1)
        elif diff == 'Diff-1':
            transformed = df[col].diff().dropna()
            treated = remove_outliers(transformed)
            # res = adf_test(transformed.dropna())
            df_trans[col] = transformed
            # df_trans = pd.concat([df_trans,treated],axis=1)
        elif diff == 'Log-1':
            transformed = np.log(df[col])#.dropna()
            treated = remove_outliers(transformed)
            # res = adf_test(transformed.dropna())
            df_trans[col] = transformed
            # df_trans = pd.concat([df_trans,treated],axis=1)
        elif diff == 'Diff-2':
            transformed = df[col].diff().diff().dropna()
            treated = remove_outliers(transformed)
            # res = adf_test(transformed.dropna())
            df_trans[col] = transformed
            # df_trans = pd.concat([df_trans,treated],axis=1)
        else:
            print(f"transformation not orderred")
            raise
        # if res == 'N' or res == 'X':
        #     return print(f"transformed data column variable stationary Adfuller Test is fail: variable name ={col},{diff},{res}")
    return df_trans

 
'''
    - pytorch dataloader default collate_fn fail pandas period_Index
'''
def load_data_timeindex(data_path):
    # load data
    # Quatery
    df_Q = pd.read_excel(data_path, index_col='date', sheet_name='df_Q', header=0)
    df_Q = df_Q.iloc[:,:].astype('float')
    df_Q.index.name = 'date'
    # df_Q = remove_outliers(df_Q)
    # Monthly
    df_M = pd.read_excel(data_path, index_col='date', sheet_name='df_M', header=0)
    df_M = df_M.iloc[:,:].astype('float')
    df_M.index.name = 'date'
    # df_M = remove_outliers(df_M)
    # Variable Info
    var_info = pd.read_excel(data_path, sheet_name='df_var_info', header=0)
    # diff transform for stationary
    df_Q_trans = transform_adf(df_Q, var_info)
    df_M_trans = transform_adf(df_M, var_info)
    return df_Q, df_Q_trans, df_M, df_M_trans, var_info


def set_lag_missing(df_trans, var_info, freq):
    df_set_lag = df_trans.copy()
    for col in df_trans.columns:
      # print(f"set_lag_column : {col}")
      lag = var_info[var_info['ID'] == col]['LAG'].values[0]
      df_set = df_trans[col]
      if lag > 0:
          df_set_lag[col] = df_set[:-(lag)]
      else:
          df_set = df_set
          df_set_lag[col] = df_set
    return df_set_lag

def save_model(epoch, lr, model, model_dir, model_name='pems08', horizon=12):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_name = os.path.join(model_dir, model_name+str(horizon)+'.bin')
    torch.save(
        {
            'epoch': epoch,
            'lr': lr,
            'model': model.state_dict(),
        }, file_name)
    print('save model in ', file_name)


def load_model(model, model_dir, model_name='pems08', horizon=12):
    if not model_dir:
        return
    file_name = os.path.join(model_dir, model_name+str(horizon)+'.bin')

    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(
            checkpoint['epoch']))
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print('loaded the model...', file_name,
              'now lr:', lr, 'now epoch:', epoch)
    return model, lr, epoch


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'dtype1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'dtype2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type-e':
        lr_adjust = {epoch: 0.2}
        if epoch <= 20:
             lr_adjust[epoch] = 0.15
        elif lr_adjust[epoch] <= 0.001:
            lr_adjust[epoch] = 0.001
        else:
            lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}

    elif args.lradj == '1':
        lr_adjust = {epoch: args.learning_rate * (0.98 ** (epoch // 1))}
        if lr_adjust[epoch] <= 0.000001:
            lr_adjust[epoch] = 0.000001

    elif args.lradj == '2':
        lr_adjust = {
            0: 0.02, 10: 0.01, 30: 0.005, 50: 0.001, 80: 0.0005, 100: 0.0001, 130: 0.00005, 150: 0.00001, 180:0.000005, 200:0.000001
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch <
                     5 else args.learning_rate*0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
    return lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, model_id):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_id)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, model_id)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, model_id):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + model_id +'_best_checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # e = 0.00000001
        self.mean = data.mean(0)
        std_s = data.std(0)
        self.std = np.where(std_s == 0, 0.000001, std_s)
        # print(f"self.std: {self.std}")

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(
            data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
    
# class MinMaxScaler:
#     def __init__(self):
#         self.max_num = -np.inf
#         self.min_num = np.inf
        
#     def fit(self, data):
#         if data is None:
#            print("fit() missing 1 required positional argument: 'X'")
#         self.max_num = np.max(data) 
#         self.min_num = np.min(data)
    
#     def transform(self,data):
#         if data is None:
#            print("fit() missing 1 required positional argument: 'X'") 
#         max_num = torch.from_numpy(self.max_num).type_as(data).to(
#             data.device) if torch.is_tensor(data) else self.max_num  
#         min_num = torch.from_numpy(self.min_num).type_as(data).to(
#             data.device) if torch.is_tensor(data) else self.min_num
#         return (data - min_num) / (max_num - min_num)
    
#     def inverse_transform(self, data):
#         if data is None:
#            print("fit() missing 1 required positional argument: 'X'") 
#         max_num = torch.from_numpy(self.max_num).type_as(data).to(
#             data.device) if torch.is_tensor(data) else self.max_num  
#         min_num = torch.from_numpy(self.min_num).type_as(data).to(
#             data.device) if torch.is_tensor(data) else self.min_num
#         return (data*(max_num - min_num)) + min_num
        
def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

# def test_params_flop(model,x_shape):
#     """
#     If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
#     """
#     model_params = 0
#     for parameter in model.parameters():
#         model_params += parameter.numel()
#         print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
#     from ptflops import get_model_complexity_info
#     with torch.cuda.device(0):
#         macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
#         # print('Flops:' + flops)
#         # print('Params:' + params)
#         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#         print('{:<30}  {:<8}'.format('Number of parameters: ', params))
