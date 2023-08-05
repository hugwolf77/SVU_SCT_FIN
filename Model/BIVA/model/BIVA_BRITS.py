import os
os.chdir('/content/SVU_SCT_FIN/Model/BIVA/model')

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
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
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))
        imputations = self.merge_ret(ret_f, ret_b)
        return imputations

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']

        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2
        # print(f"ret_f['imputations']: \n {ret_f['imputations']}")
        # print(f"ret_b['imputations']:\n {ret_b['imputations']}")
        # print(f"imputations: \n {imputations}")
        # raise

        ret_f['loss'] = loss
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

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
            
        # print(f"key : {key}")
        # print(f"ret[key]:\n {ret[key][0][:,0]} {ret[key].shape}")
        # raise

        return ret
