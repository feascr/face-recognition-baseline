from torch import nn
import torch.nn.functional as F

from . import register_head


@register_head('fc-head')
class FCHead(nn.Module):
    def __init__(self, embedding_size=128, num_classes=100):
        super(FCHead, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, embeddings, targets):
        return self.fc(embeddings)