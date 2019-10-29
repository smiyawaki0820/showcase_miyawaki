import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

from pytorch_memlab import profile


class BiGRUForSRL(nn.Module):
    # def __init__(self, dim_in: int, dim_u: int, num_layers: int):
    def __init__(self, dim_in: int, dim_u: int, num_layers: int, dropout: int):
        super(BiGRUForSRL, self).__init__()

        self.dim_in = dim_u
        self.dim_u = dim_u
        self.depth = num_layers
        self.embed_dim = dim_u
        self.num_head = 1

        # input is a set of embedding and predicate marker
        self.gru_in = nn.GRU(dim_in, dim_u, num_layers=1, batch_first=True)

        ## layer 数分のiteration #*2
        self.grus = nn.ModuleList([nn.GRU(dim_u, dim_u, num_layers=1, batch_first=True) for _ in range(num_layers - 1)])
        self.gate_linear_prev = nn.Linear(dim_u, 1)
        self.gate_linear_now = nn.Linear(dim_u, 1)
        self.gate_sigmoid = nn.Sigmoid()

    # x = input
    #@profile
    def forward(self, x, each_layer_in, iter_num):
        if torch.cuda.is_available():
            x = x.cuda()
        #print('in bi_gru', self.dim_in, self.dim_u, x.size())
        out, _ = self.gru_in(x)
        each_layer_output = []
        if iter_num <= 0:
            for gru in self.grus:
                flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)
                output, _ = gru(flipped)
                out = flipped + output
                each_layer_output.append(self.reverse(out.transpose(0, 1)).transpose(0, 1)) # output_hidden が格納された list
        else:
            # self.grus : Module_lis
            for i, gru in enumerate(self.grus):
                flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)
                
                if torch.cuda.is_available():
                    layer_i = autograd.Variable(each_layer_in[i]).cuda()
                else:
                    layer_i = autograd.Variable(each_layer_in[i]) 

                ## gate による hat{h}
                gate_lambda = self.gate_sigmoid(self.gate_linear_prev(layer_i) + self.gate_linear_now(flipped))
                gate_output = gate_lambda * layer_i + (1 - gate_lambda) * (flipped)
                output, _ = gru(gate_output)    # output/out.size = [batch, sent_len, dim_u]  # batch ??

                out = flipped + output
                each_layer_output.append(self.reverse(out.transpose(0, 1)).transpose(0, 1).cpu()) # output_hidden が格納された list

        return self.reverse(out.transpose(0, 1)).transpose(0, 1), each_layer_output     # scores, hidden_list #

    def reverse(self, x):
        idx = torch.arange(x.size(0) - 1, -1, -1).long()
        idx = torch.LongTensor(idx)
        if torch.cuda.is_available():
            idx = idx.cuda()
        return x[idx]

