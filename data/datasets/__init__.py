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

DATASETS_REGISTRY = {}

def create_dataset(name, *args, **kwargs):
    assert name in DATASETS_REGISTRY, 'uknown dataset name'
    return DATASETS_REGISTRY[name](*args, **kwargs)

def register_dataset(name):
    def register_dataset_constructor(backbone_constructor):
        if name in DATASETS_REGISTRY:
            raise ValueError('Cannot register duplicate dataset {}'.format(name))
        DATASETS_REGISTRY[name] = backbone_constructor
        return backbone_constructor

    return register_dataset_constructor

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'data.datasets')


def registered_datasets():
    return sorted(DATASETS_REGISTRY.keys())