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

HEAD_REGISTRY = {}

def create_head(name, *args, **kwargs):
    assert name in HEAD_REGISTRY, 'uknown head model'
    return HEAD_REGISTRY[name](*args, **kwargs)

def register_head(name):
    def register_head_constructor(head_constructor):
        if name in HEAD_REGISTRY:
            raise ValueError('Cannot register duplicate model {}'.format(name))
        HEAD_REGISTRY[name] = head_constructor
        return head_constructor

    return register_head_constructor

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'models.heads')


def registered_backbones():
    return sorted(HEAD_REGISTRY.keys())