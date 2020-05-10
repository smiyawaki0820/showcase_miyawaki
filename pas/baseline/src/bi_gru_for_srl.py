import torch
import torch.nn as nn


class BiGRUForSRL(nn.Module):
    # def __init__(self, dim_in: int, dim_u: int, num_layers: int):
    def __init__(self, dim_in: int, dim_u: int, num_layers: int, dropout: int):
        super(BiGRUForSRL, self).__init__()

        self.dim_in = dim_u
        self.dim_u = dim_u
        self.depth = num_layers

        # input is a set of embedding and predicate marker
        self.gru_in = nn.GRU(dim_in, dim_u, num_layers=1, batch_first=True)
        self.grus = nn.ModuleList([nn.GRU(dim_u, dim_u, num_layers=1, batch_first=True) for _ in
                                   range(num_layers - 1)])

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        #print('in bi_gru', self.dim_in, self.dim_u, x.size())
        out, _ = self.gru_in(x)
        for gru in self.grus:
            flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)
            output, _ = gru(flipped)
            out = flipped + output

        return self.reverse(out.transpose(0, 1)).transpose(0, 1)

    def reverse(self, x):
        idx = torch.arange(x.size(0) - 1, -1, -1).long()
        idx = torch.LongTensor(idx)
        if torch.cuda.is_available():
            idx = idx.cuda()
        return x[idx]
