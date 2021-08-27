import torch
import torch.nn as nn
from functools import wraps


def timer(fn):
    input_count = 1
    @wraps(fn)
    
    def inner(*args, **kwargs):
        nonlocal input_count

        from time import perf_counter
        start = perf_counter()
        result = fn(*args, **kwargs)
        end = perf_counter()
        elapsed = end - start

        print('\n\n{}() took {:.8f}s time to run for input: {}'.format(fn.__name__, elapsed, input_count))
        input_count+=1
        return fn(*args, **kwargs)
    
    return inner

        

class ThreeNodeGraph:
    """
    This class implements a 3 node graph containing convolutional_output, 
    maxpool output and Softmax Output, run on an input.
    """
    def __init__(self, input):
        self.input = input
    
    @timer
    def three_node_graph(self):
        """
        This method is responsible for running 3 node graph on a input.

        Returns:
            input, convolutional_output, maxpool output and Softmax Output
        """
        
        self.conv_layer = nn.Conv2d(in_channels=3, out_channels=6, padding=0, kernel_size=3, stride=1, bias=True)
        self.conv_out = self.conv_layer(self.input)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.max_pool_out = self.max_pool(self.conv_out)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_out = self.softmax(self.max_pool_out)

        return self.input, self.conv_out, self.max_pool_out, self.softmax_out

    def __call__(self):
        try:
            model_detail = 'Input is : {} \n \n Conv is : {} \n \n Maxpool is : {} \n \n Softmax Output is {} '.format(self.input.data, self.conv_out.data, self.max_pool_out.data, self.softmax_out.data) 
        except AttributeError:
            model_detail = 'Input is : {} \n\n\n'.format(self.input)
        return model_detail


        

for i in range(10):
    input = torch.rand(1, 3, 4, 4)
    graph_obj = ThreeNodeGraph(input)
    graph_obj.three_node_graph()
    print(graph_obj())
