import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math
from bi_gru_packed import BiGRUForSRL
import ipdb

max_sentence_length = 90

# PackedE2EStackedBiRNN(nn.Module)
# https://github.com/cl-tohoku/showcase-konno/blob/master/src/jp_pas/models.py#L112-L152
class E2EStackedBiRNN(nn.Module):
    def __init__(self, dim_u: int, depth: int,
                 dim_out: int,
                 embedding_matrix,
                 drop_u: int,
                 fixed_word_vec: bool,
                 iter_num: int, threshold: float,
                 null_label: str):
        super(E2EStackedBiRNN, self).__init__()

        self.vocab_size = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)
        self.hidden_dim = dim_u
        self.depth = depth
        self.iter_num = iter_num
        self.threshold = threshold
        self.null_label = null_label

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.word_emb.weight = nn.Parameter(embedding_matrix)
        if fixed_word_vec:
            self.word_emb.weight.requires_grad = False

        ### BiGRU ###
        self.gru = BiGRUForSRL(self.embedding_dim + 1+4, dim_u, num_layers=depth, dropout=drop_u)
        self.output_layer = nn.Linear(dim_u, dim_out)

    def forward(self, x, temp, init=None):
        words, is_target, xs_len = x

        if temp.size(0) == 0:
            temp = torch.zeros(words.size(0), words.size(1), 4)

        if torch.cuda.is_available():
            words = autograd.Variable(words).cuda()
            is_target = autograd.Variable(is_target).cuda()
            temp = autograd.Variable(temp).cuda()
        else:
            words = autograd.Variable(words)
            is_target = autograd.Variable(is_target)
            temp = autograd.Variable(temp)

        try:
            embeds = self.word_emb(words)
        except RuntimeError:
            embeds = self.word_emb(words.cpu())
            if torch.cuda.is_available():
                embeds = autograd.Variable(embeds).cuda()
            else:
                embeds = autograd.Variable(embeds)

        # Iter 部分
        scores = []
        pass_num = [0,0,0]
        for t in range(self.iter_num):
            inputs = torch.cat([embeds, is_target, temp], dim=2)
            outputs = self.gru(inputs, xs_len, t) # ここで BiGRU 部分に渡す
            res = [F.softmax(self.output_layer(out)) for out in outputs]
            scores.append([F.log_softmax(self.output_layer(out[:int(x_len)]), dim=1) for out, x_len in zip(outputs, xs_len)])
            temp = torch.stack([r * (r > self.threshold).float() for r in res])
            if self.null_label == "inc": 
                pass
            elif self.null_label == "exc":
                temp[:,:,-1] = 0
            pass_num[t] = int(torch.sum((temp > 0)))

        #return [F.log_softmax(self.output_layer(out[:int(x_len)]), dim=1) for out, x_len in zip(outputs, xs_len)]
        if init:
            #return [F.log_softmax(self.output_layer(out[:int(x_len)]), dim=1) for out, x_len in zip(outputs, xs_len)], pass_num
            return [F.softmax(self.output_layer(out)) for out in outputs], pass_num
        else:
            return scores, pass_num
