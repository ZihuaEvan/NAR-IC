import torch
import torch.nn as nn
import pickle
from lib.config import cfg

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, logit, target_seq):
        logit = logit.view(-1, logit.shape[-1])
        target_seq = target_seq.view(-1)
        loss = self.criterion(logit, target_seq)
        return loss