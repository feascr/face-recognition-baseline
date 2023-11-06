import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Iterator
from torch.nn import functional as F

from .heads import create_head
from .backbones import create_backbone
from . import register_model


@register_model('simple_representation_learner')
class SimpleRepresentationLearner(nn.Module):
    def __init__(
            self,
            backbone,
            head,
            backbone_kwargs={},
            head_kwargs={}
        ):
        super(SimpleRepresentationLearner, self).__init__()
        self.backbone = create_backbone(backbone, **backbone_kwargs)
        self.head = create_head(head, **head_kwargs)

    def forward_embeddings(self, x):
        return self.backbone(x)

    def forward(self, x, targets):
        embeddings = self.forward_embeddings(x)
        logits = self.head(embeddings, targets)
        return logits
    
    def backbone_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Returns an iterator over module parameters excluding head module.
        This is typically passed to an optimizer.
        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            Parameter: module parameter
        Example::
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
        """
        for name, param in self.named_parameters(recurse=recurse):
            if 'head' in name:
                continue
            yield param
    
    def head_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            if 'head' in name:
                yield param