
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.autograd import Variable
# from torch.nn.parameter import Parameter

# import numpy as np
# import math

from model import BIVA_BRITS as BRITS
from model import BIVA_LSTM_VAE as LSTM_VAE


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size,
                                stride=stride, padding=0)
    def forward(self, x):
        # padding on the both ends of time series

        if self.kernel_size % 2 != 0:  # even must be modify
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        else:
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size) // 2, 1)

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
        self.moving_avg = [moving_avg(kernel, stride=1)
                           for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean

class RIN(nn.Module):
    def __init__(self):
        super(RIN, self).__init__()
        self.build()

    def build(self):
        self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

    def set_RIN(self, x):
        # print('/// RIN ACTIVATED ///\r', end='')
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        x = x * self.affine_weight + self.affine_bias
        return x

    def off_RIN(self, x):
        x = x - self.affine_bias
        x = x / (self.affine_weight + 1e-10)
        stdev = stdev[:, :, -1:]
        means = means[:, :, -1:]
        x = x * stdev
        x = x + means
        return x


class Model(nn.Module):
    """
    Bi-direction Recurrent Imputation & VAE
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len          # manthly observation input value
        self.pred_len = args.pred_len        # qurter state target value
        self.label_len = args.label_len

        self.channels = args.channels        # times series or feature
        self.latent_size = args.vae_latent_size
        self.infer_hid_size = args.infer_hid_size

        self.conv_kernal = args.conv_kernal  # if use conv1d layer
        # time Series decompose average pooling kernel size
        self.kernel_size = args.moving_avg

        self.batch_size = args.batch_size
        self.conv1d = args.conv1d
        self.RIN = args.RIN                  # boolen, Reverse Instance Normalize option
        self.combination = args.combination  # boolen, compose time Series part option
        
        self.build(args)

    def build(self, args):
        # Decompose
        if isinstance(self.kernel_size, list):
            self.decomposition = series_decomp_multi(self.kernel_size)
        else:
            self.decomposition = series_decomp(self.kernel_size)
            
        # self.log_softmax = F.log_softmax()
        # self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

        # brits
        self.BRITS = BRITS.Model(args)
        # lstm_vae
        self.LSTM_VAE = LSTM_VAE.Model(args)

        # Reverse Instance Normalize & T-S combination param
        if self.RIN:
            self.RIN_func = RIN()

        if self.combination:
            self.alpha = nn.Parameter(torch.ones(1, 1, 1))

        if self.conv1d:
            # self.Conv1d_Seasonal = nn.Conv1d(
            #     self.latent_size*2, 1, kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)
            
            self.Conv1d_Seasonal_1 = nn.Conv1d(
                self.latent_size*2, self.latent_size, kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)
            self.Conv1d_Seasonal_2 = nn.Conv1d(
                self.latent_size, int(self.latent_size/2), kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)
            self.Conv1d_Seasonal_3 = nn.Conv1d(
                int(self.latent_size/2), 1, kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)            
            
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

            # self.Conv1d_Trend = nn.Conv1d(
            #     self.channels, 1, kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)

            self.Conv1d_Trend_1 = nn.Conv1d(
                self.channels, int(self.channels/4), kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)
            self.Conv1d_Trend_2 = nn.Conv1d(
                int(self.channels/4), int(self.channels/16), kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)
            self.Conv1d_Trend_3 = nn.Conv1d(
                int(self.channels/16), 1, kernel_size=self.conv_kernal, dilation=1, stride=1, groups=1)

            
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend.weight = nn.Parameter(
                (1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend.weight = nn.Parameter(
                (1/self.seq_len)*torch.ones([self.pred_len, self.seq_len]))

    def loss_function(self, recons, input, mu, log_var):
        """
        Computes the VAE loss function.
        """
        kld_weight = 0.5  # Account for the minibatch samples from the dataset
        recons_weight = 0.5
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean( -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0 )
        loss = recons_weight * recons_loss  + kld_weight * kld_loss
        return torch.sum(loss), recons_loss, -kld_loss

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.RIN:
            x = self.RIN_func.set_RIN(x)
        # BRITS - imputation
        x = self.BRITS(x)
        imputed_x = x['imputations']
        imputed_loss = x['loss']
        # print(f"BRITS_out_x.shape: {x.shape}")

        # decompose timeseries
        seasonal_init, trend_init = self.decomposition(imputed_x)
        # -- Trend --
        trend_init = trend_init.permute(0, 2, 1)
        trend_output_1 = self.Conv1d_Trend_1(trend_init)
        trend_output_2 = self.Conv1d_Trend_2(trend_output_1)
        trend_output_3 = self.Conv1d_Trend_3(trend_output_2)
        trend_output = self.Linear_Trend(trend_output_3)
        # print(f"trend_output.shape: {trend_output.shape}")

        # -- Seasonal --
        # LSTM_VAE
        recon_output, mu, logvar, seasonal_enc_output_x, seasonal_output_z = self.LSTM_VAE(seasonal_init)
        VAE_loss,_,_ = self.loss_function(recon_output, seasonal_init, mu, logvar)
        
        # print(f"seasonal_output_z.shape: {seasonal_output_z.shape}")
        # print(f"seasonal_enc_output_x.shape: {seasonal_enc_output_x.shape}")
        # print(f"recon_output.shape: {recon_output.shape}")
        # raise
        #
        seasonal_enc_output_x = seasonal_enc_output_x.permute(0, 2, 1) 
        seasonal_output_z = seasonal_output_z.permute(0, 2, 1)
        seasonal_output_x_z = torch.cat([seasonal_enc_output_x,seasonal_output_z],dim=1)
        
       
        # seasonal_output = self.Conv1d_Seasonal(seasonal_output_x_z)
        seasonal_output_1 = self.Conv1d_Seasonal_1(seasonal_output_x_z)
        seasonal_output_2 = self.Conv1d_Seasonal_2(seasonal_output_1)
        seasonal_output_3 = self.Conv1d_Seasonal_3(seasonal_output_2)
        
        seasonal_output = self.Linear_Seasonal(seasonal_output_3)

        if self.combination:
            states = (seasonal_output*(self.alpha)) + (trend_output*(1-self.alpha))
        else:
            states = seasonal_output + trend_output

        states = states.permute(0, 2, 1)  # to [Batch, Output length, Channel]

        if self.RIN:
            states = self.RIN_func.off_RIN(states)

        return states, VAE_loss, recon_output, seasonal_init, imputed_loss, imputed_x
