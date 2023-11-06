from pathlib import Path
import os
import sys
import importlib


def import_all_modules(root, base_module):
    for file in os.listdir(root):
        if file.endswith(('.py', '.pyc')) and not file.startswith('_'):
            module = file[: file.find('.py')]
            if module not in sys.modules:
                module_name = '.'.join([base_module, module])
                importlib.import_module(module_name)

BACKBONES_REGISTRY = {}

def create_backbone(name, *args, **kwargs):
    assert name in BACKBONES_REGISTRY, 'uknown backbone name'
    return BACKBONES_REGISTRY[name](*args, **kwargs)

def register_backbone(name):
    def register_backbone_constructor(backbone_constructor):
        if name in BACKBONES_REGISTRY:
            raise ValueError('Cannot register duplicate model {}'.format(name))
        BACKBONES_REGISTRY[name] = backbone_constructor
        return backbone_constructor
    return register_backbone_constructor

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'models.backbones')


def registered_backbones():
    return sorted(BACKBONES_REGISTRY.keys())