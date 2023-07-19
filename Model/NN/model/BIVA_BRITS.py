import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import BIVA_RITS as RITS  # RITS


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.rnn_hid_size = args.rnn_hid_size

        self.build(args)

    def build(self, args):
        self.rits_f = RITS.Model(args)
        self.rits_b = RITS.Model(args)

    def forward(self, data):
        ret_f = self.rits_f(data)
        ret_b = self.reverse(self.rits_b(data))
        imputations = self.merge_ret(ret_f, ret_b)
        return imputations

    def merge_ret(self, ret_f, ret_b):
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        return imputations

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret
