import torch.nn as nn
import torch

def timer(fn):
    def inner(*args, **kwargs):
        from time import perf_counter
        start = perf_counter()
        result = fn(*args, **kwargs)
        end = perf_counter()
        elapsed = end - start
        print('{}() took {:.8f}s '.format(fn.__name__, elapsed))
        return result
    return inner

class Softmax:
    def __init__(self, input, dim):
        self.input = input
        self. dim = dim

    def calculate_softmax(self):
        return nn.Softmax(dim=1)(self.input)


