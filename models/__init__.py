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

MODELS_REGISTRY = {}

def create_model(name, *args, **kwargs):
    assert name in MODELS_REGISTRY, 'uknown model name'
    return MODELS_REGISTRY[name](*args, **kwargs)

def register_model(name):
    def register_model_constructor(backbone_constructor):
        if name in MODELS_REGISTRY:
            raise ValueError('Cannot register duplicate model {}'.format(name))
        MODELS_REGISTRY[name] = backbone_constructor
        return backbone_constructor

    return register_model_constructor

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'models')


def registered_models():
    return sorted(MODELS_REGISTRY.keys())