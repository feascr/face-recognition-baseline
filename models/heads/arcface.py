import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from . import register_head


@register_head('arcface')
class ArcFace(nn.Module):
    def __init__(self, embedding_size=128, num_classes=100, s=64.0, m=0.50):
        super(ArcFace, self).__init__()

        self._embedding_size = embedding_size
        self._num_classes = num_classes
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, targets):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > 0, phi, cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        logits_target = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        logits_target *= 64
        return logits_target