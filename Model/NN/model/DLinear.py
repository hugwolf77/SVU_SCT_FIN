from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# from layers.Gru import *

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series

        if self.kernel_size%2 != 0: ##  even must be modify
          front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
          end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        else:
          front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
          end = x[:, -1:, :].repeat(1, (self.kernel_size ) // 2, 1)

        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 

class Model(nn.Module):
    """
    DLinear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomp
        self.kernel_size = configs.moving_avg
        if isinstance(self.kernel_size, list):
            self.decomposition = series_decomp_multi(self.kernel_size)
        else:
            self.decomposition = series_decomp(self.kernel_size)

        self.channels = configs.enc_in
        self.label_len = configs.label_len

        self.conv1d = configs.conv1d
        # self.conv1d_activation = configs.conv1d_activation

        # self.pred_lin_add = configs.pred_lin_add

        # self.Lin_activation = configs.Lin_activation

        self.RIN = configs.RIN
        self.combination =  configs.combination


        # self.RNN_lyr = configs.RNN_lyr
        # self.RNN_stack = configs.rnn_stack
        # self.hidden_dim = configs.hidden
        # self.dropout = configs.dropout

        self.conv_kernal = configs.conv_kernal

        self.batch_size = configs.batch_size

        if self.conv1d:
          self.Conv1d_Seasonal = nn.Conv1d(self.channels, self.label_len, kernel_size=self.conv_kernal, dilation=1, stride=1, groups= 1)
          self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
          self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
         
          self.Conv1d_Trend = nn.Conv1d(self.channels, self.label_len, kernel_size=self.conv_kernal, dilation=1, stride=1, groups= 1)
          self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
          self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        else:
          self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
          self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
          self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
          self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        # if self.pred_lin_add:
        #   self.Linear_Seasonal_add = nn.Linear(self.seq_len,self.seq_len)
        #   self.Linear_Seasonal_add.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.seq_len,self.seq_len]))
        #   self.Linear_Trend_add = nn.Linear(self.seq_len,self.seq_len)
        #   self.Linear_Trend_add.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.seq_len,self.seq_len]))

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

        ### combination ###
        if self.combination:
          self.alpha = nn.Parameter(torch.ones(1,1,1))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.RIN:
            print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x * self.affine_weight + self.affine_bias

        seasonal_init, trend_init = self.decomposition(x)

        if self.conv1d :
          seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
          seasonal_output = self.Conv1d_Seasonal(seasonal_init)
          trend_output = self.Conv1d_Trend(trend_init)

          # if self.conv1d_activation :
          #   seasonal_output = F.relu(seasonal_output)
          #   trend_output = F.relu(trend_output)

          # if self.pred_lin_add:
          #   seasonal_output = self.Linear_Seasonal_add(seasonal_output)
          #   trend_output = self.Linear_Trend_add(trend_output)

          # if self.Lin_activation:
          #   seasonal_ouput = F.relu(seasonal_output)
          #   trend_output = F.relu(trend_output)

          seasonal_output = self.Linear_Seasonal(seasonal_output)
          trend_output = self.Linear_Trend(trend_output)

          # if self.Lin_activation:
          #   seasonal_ouput = F.relu(seasonal_output)
          #   trend_output = F.relu(trend_output)
          
        else:
          seasonal_output, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
          seasonal_output = self.Linear_Seasonal(seasonal_output)
          trend_output = self.Linear_Trend(trend_init)

        if self.combination:
          x = (seasonal_output*(self.alpha)) + (trend_output*(1-self.alpha))
          # print(f"combination_alpha trend: {(1-self.alpha)}")
          # print(f"combination_alpha seasonal: {(self.alpha)}")
        
        else:
          x = seasonal_output + trend_output

        x = x.permute(0,2,1) # to [Batch, Output length, Channel]

        if self.RIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            stdev = stdev[:,:,-1:]
            means = means[:,:,-1:]
            x = x * stdev
            x = x + means

        return x 

