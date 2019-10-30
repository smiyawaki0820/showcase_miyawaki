import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math
from bi_gru_edit import BiGRUForSRL
from pytorch_memlab import profile
import ipdb
import numpy as np

max_sentence_length = 90


class E2EStackedBiRNN(nn.Module):
    def __init__(self, dim_u: int, depth: int,
                 dim_out: int,
                 embedding_matrix,
                 drop_u: int,
                 fixed_word_vec: bool):
        super(E2EStackedBiRNN, self).__init__()

        self.vocab_size = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)
        self.hidden_dim = dim_u
        self.depth = depth
        self.label_dim = 32

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.word_emb.weight = nn.Parameter(embedding_matrix)
        if fixed_word_vec:
            self.word_emb.weight.requires_grad = False

        # input is a set of embedding and predicate marker
        self.gru = BiGRUForSRL(self.embedding_dim + 1+32, dim_u, num_layers=depth, dropout=drop_u)
        self.output_layer = nn.Linear(dim_u, dim_out)
        self.label_weight = nn.Linear(dim_out, self.label_dim, bias=False)


    def forward(self, x, temp, it, thres):
        words, is_target = x
        #print("words: ", words.size())
        if torch.cuda.is_available():
            words = autograd.Variable(words).cuda()
            is_target = autograd.Variable(is_target).cuda()
        else:
            words = autograd.Variable(words)
            is_target = autograd.Variable(is_target)

        embeds = self.word_emb(words)
        #each_layer_in = [torch.zeros([words.size()[0], words.size()[1], self.hidden_dim]) for _ in range(self.depth)]
        for t in range(it):
            if torch.cuda.is_available():
                temp = autograd.Variable(temp).cuda()
            else:
                temp = autograd.Variable(temp)
            inputs = torch.cat([embeds, is_target, temp], dim=2)
            outputs = self.gru(inputs)
            #each_layer_in = each_layer_out
            res = []
            for out in outputs:
                hoge = F.log_softmax(self.output_layer(out))
                hoge[:,-1] = 0.0
                res.append(F.log_softmax(self.label_weight(hoge)))
            temp = torch.stack(res)

        try:
            return [F.log_softmax(self.output_layer(out)) for out in outputs]
        except:
            ipdb.set_trace()

    def filtering(self, res, temp, thres):
        for batch_idx, batch in enumerate(res):
            batch_high_score = {}
            for word_idx in range(batch.size(0)):
                if torch.max(batch[word_idx]) >= math.log(thres):
                    batch_high_score[word_idx] = batch[word_idx]
                    temp[batch_idx, word_idx, :] = batch[word_idx]
        return temp
