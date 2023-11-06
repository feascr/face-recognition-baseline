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

SAMPLERS_REGISTRY = {}

def create_sampler(name, *args, **kwargs):
    assert name in SAMPLERS_REGISTRY, 'uknown sampler name'
    return SAMPLERS_REGISTRY[name](*args, **kwargs)

def register_sampler(name):
    def register_sampler_constructor(backbone_constructor):
        if name in SAMPLERS_REGISTRY:
            raise ValueError('Cannot register duplicate sampler {}'.format(name))
        SAMPLERS_REGISTRY[name] = backbone_constructor
        return backbone_constructor

    return register_sampler_constructor

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'data.samplers')


def registered_samplers():
    return sorted(SAMPLERS_REGISTRY.keys())