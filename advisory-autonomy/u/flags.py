
import os

modules = ['numpy', 'pandas', 'scipy', 'matplotlib', 'torch', 'sklearn', 'visdom']
for name in modules:
    globals()[name] = name not in ('sklearn', 'visdom')

def none():
    for name in modules:
        globals()[name] = False

def include(*args):
    for name in args:
        assert name in modules
        globals()[name] = True

def exclude(*args):
    for name in args:
        assert name in modules
        globals()[name] = False

def only(*args):
    for name in args:
        assert name in modules
    for name in modules:
        globals()[name] = name in args