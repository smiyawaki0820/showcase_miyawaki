import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from bi_gru_for_srl import BiGRUForSRL

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

        self.word_emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.word_emb.weight = nn.Parameter(embedding_matrix)
        if fixed_word_vec:
            self.word_emb.weight.requires_grad = False

        # input is a set of embedding and predicate marker
        self.gru = BiGRUForSRL(self.embedding_dim + 1+4, dim_u, num_layers=depth, dropout=drop_u)
        self.output_layer = nn.Linear(dim_u, dim_out)

    def forward(self, x, l):
        words, is_target = x
        
        if torch.cuda.is_available():
            words = autograd.Variable(words).cuda()
            is_target = autograd.Variable(is_target).cuda()
        else:
            words = autograd.Variable(words)
            is_target = autograd.Variable(is_target)

        embeds = self.word_emb(words)
        #print(embeds.size(), is_target.size(), l.size())
        inputs = torch.cat([embeds, is_target, l], dim=2)
        
        outputs = self.gru(inputs)
        # sm = nn.Softmax()
        #for out in outputs:
        #  print('## out ##', self.output_layer(out))
        ret = [F.log_softmax(self.output_layer(out)) for out in outputs]
        # print('## ret ##', ret, sep='\n')
        return ret
        
