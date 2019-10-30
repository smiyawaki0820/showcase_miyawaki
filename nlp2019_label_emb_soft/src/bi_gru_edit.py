import torch
import torch.nn as nn
from torch import autograd
from pytorch_memlab import profile
import ipdb

class BiGRUForSRL(nn.Module):
    # def __init__(self, dim_in: int, dim_u: int, num_layers: int):
    def __init__(self, dim_in: int, dim_u: int, num_layers: int, dropout: int):
        super(BiGRUForSRL, self).__init__()

        self.dim_in = dim_u
        self.dim_u = dim_u
        self.depth = num_layers

        # input is a set of embedding and predicate marker
        self.gru_in = nn.GRU(dim_in, dim_u, num_layers=1, batch_first=True)
        
        ## layer 数分のiteration #*2
        self.each_layer_out = [nn.GRU(dim_u, dim_u, num_layers=1, batch_first=True) for _ in range(num_layers - 1)]
        self.grus = nn.ModuleList(self.each_layer_out)

    # x = input
    def forward(self, x): #, each_layer_in):
        if torch.cuda.is_available():
            x = x.cuda()
        #print('in bi_gru', self.dim_in, self.dim_u, x.size())
        try:
            out, _ = self.gru_in(x)
        except:
            ipdb.set_trace()
        #each_layer_output = []
        # self.grus : Module_lis
        for i, gru in enumerate(self.grus):
            flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)
            #if torch.cuda.is_available():
            #    layer_i = autograd.Variable(each_layer_in[i]).cuda()
            #else:
            #    layer_i = autograd.Variable(each_layer_in[i])
            output, _ = gru(flipped)
            #output, _ = gru(torch.cat([flipped, layer_i], dim=2))    # output/out.size = [batch, sent_len, dim_u]  # batch ??
            out = flipped + output
            #each_layer_output.append(self.reverse(out.transpose(0, 1)).transpose(0, 1))

        return self.reverse(out.transpose(0, 1)).transpose(0, 1) #, each_layer_output

    def reverse(self, x):
        idx = torch.arange(x.size(0) - 1, -1, -1).long()
        idx = torch.LongTensor(idx)
        if torch.cuda.is_available():
            idx = idx.cuda()
        return x[idx]
