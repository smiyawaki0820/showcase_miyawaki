import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# PackedBiGRUForSRL
# https://github.com/cl-tohoku/showcase-konno/blob/master/src/jp_pas/models.py#L200-L252
class BiGRUForSRL(nn.Module):
    def __init__(self, dim_in: int, dim_u: int, num_layers: int, dropout: int):
        super(BiGRUForSRL, self).__init__()

        self.dim_in = dim_u
        self.dim_u = dim_u
        self.depth = num_layers
        self.embed_dim = dim_u
        self.num_head = 1

        ### GRU ###
        self.gru_in = nn.GRU(dim_in, dim_u, num_layers=1, batch_first=True)
        self.grus = nn.ModuleList([nn.GRU(dim_u, dim_u, num_layers=1, batch_first=True) for _ in range(num_layers - 1)])


    def forward(self, x, seq_size, iter_num):
        if torch.cuda.is_available():
            x = x.cuda()

        ### padding ###
        packed_x = pack_padded_sequence(x, seq_size, batch_first=True, enforce_sorted=False) # 各tensor をまとめる
        try:
            packed_out, _ = self.gru_in(packed_x)
        except RuntimeError:
            packed_out, _ = self.gru_in(packed_x.cpu())
        out, _ = pad_packed_sequence(packed_out, batch_first=True) # 各tensor に分割

        for gru in self.grus:
            flipped = self.reverse_packed(out, seq_size)
            #flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)

            ### padding ###
            packed_flipped = pack_padded_sequence(flipped, seq_size, batch_first=True, enforce_sorted=False)  # gather
            packed_output, _ = gru(packed_flipped)
            output, _ = pad_packed_sequence(packed_output, batch_first=True) # separate

            out = flipped + output

        return self.reverse_packed(out, seq_size)
        #return self.reverse(out.transpose(0, 1)).transpose(0, 1)

    def reverse_packed(self, x, xs_len):
        ids = torch.cat([torch.cat([torch.arange(x_len - 1, -1, -1), torch.arange(x_len, x.size(1))]) + i * x.size(1) for i, x_len in enumerate(xs_len)]).long()
        cat_x = x.reshape(x.size(0) * x.size(1), x.size(2))
        flipped_x = cat_x[ids]
        return flipped_x.reshape(x.shape)

    #def reverse(self, x):
    #    idx = torch.arange(x.size(0) - 1, -1, -1).long()
    #    idx = torch.LongTensor(idx)
    #    if torch.cuda.is_available():
    #        idx = idx.cuda()
    #    return x[idx]


