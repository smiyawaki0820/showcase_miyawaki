import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#https://huggingface.co/transformers/model_doc/bert.html#transformers.BertForSequenceClassification
from transformers import (
        BertModel, 
        BertForSequenceClassification, 
        BertForMaskedLM
        )
from transformers.tokenization_utils import BatchEncoding

PRE_TRAINED_MODEL_NAME = 'cl-tohoku/bert-base-japanese-char-whole-word-masking'
sys.path.append(os.getcwd())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class SABert(nn.Module):

    def __init__(self,
            dataloader,
            n_classes: int=3
            ):
        super().__init__()
        self.dataloader = dataloader
        self.bert = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=n_classes)
        #self.drop = nn.Dropout(p=0.1)
        #self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):

        """ using BertModel
        last_hidden, pooled_output = self.bert(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                )
        pooled_output = self.drop(pooled_output)
        logits = F.log_softmax(self.linear(pooled_output), dim=-1)
        return logits
        """

        # using BertForSequenceClassification
        loss, logits = self.bert(
                input_ids=input_ids,
                labels=labels
                )
        return loss, logits

    def fix_parameters(self, requires_grad=True):
        for param in self.bert.parameters():
            param.requires_grad = requires_grad

