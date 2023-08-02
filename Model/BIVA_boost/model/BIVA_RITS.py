import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - \
            torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.seq_len = args.seq_len
        self.input_size = args.channels 
        self.rnn_hid_size = args.rnn_hid_size

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.input_size*2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(
            input_size=self.input_size, output_size=self.rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(
            input_size=self.input_size, output_size=self.input_size, diag=True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)

        self.weight_combine = nn.Linear(
            self.input_size * 2, self.input_size)

        # self.delta_calc()

    def delta_calc(self, masks):
        # just 1step is time-strimp 1 month step
        delta = torch.zeros_like(masks)
        # print(f"delta.shape : {delta.shape}")
        for b in range(masks.shape[0]):
          for t in range(masks.shape[1]):
              for d in range(masks.shape[2]):
                  if t == 0:
                      delta[b,t,d] = 0
                  elif masks[b,t-1,d] == 0 and t > 0:
                      delta[b,t,d] = delta[b,t,d-1]+t-(t-1)
                  elif masks[b,t,d-1] == 1 and t > 0:
                      delta[b,t,d] = t - (t-1)
        return delta

    def forward(self, data):

        # make function for each section data

        values = data
        masks = torch.logical_not(torch.isnan(data)).float()
        # print(f"masks.shape: {masks.shape}")
        # just 1step is time-strimp 1 month step
        deltas = self.delta_calc(masks)

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        imputations = []

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h
            x_h = self.hist_reg(h)

            x_c = torch.nan_to_num(m * x) + (1 - m) * x_h
            z_h = self.feat_reg(x_c)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            c_h = alpha * z_h + (1 - alpha) * x_h
            c_c = torch.nan_to_num(m * x) + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)

        return {'imputations': imputations}
