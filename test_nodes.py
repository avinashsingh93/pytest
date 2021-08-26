import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from Node import *




class TestConvolution:

    @timer
    def test_invalid_argument_conv1d(self):
        input = torch.rand(1, 1, 10)
        with pytest.raises(TypeError):   
            layer = nn.Conv1d(in_channels=1, out_channels=1, padding=0, kernel_size=3, stride=1, bias=True, device=True)         
            layer(input)

    @timer
    def test_invalid_argument_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        with pytest.raises(TypeError):   
            layer = nn.Conv2d(in_channels=3, out_channels=6, padding=0, kernel_size=3, stride=1, bias=True, dtype=torch.float32)         
            layer(input)

    @timer
    def test_invalid_argument_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        with pytest.raises(TypeError):
            layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding=0, kernel_size='size', stride=2, arg=True)            
            layer(input) 
                 
    @timer
    def test_string_kernel_conv1d(self):
        input = torch.rand(1, 1, 10)
        with pytest.raises(TypeError):   
            layer = nn.Conv1d(in_channels=1, out_channels=1, padding=0, kernel_size='a', stride=1, bias=True)         
            layer(input)

    @timer
    def test_string_kernel_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        with pytest.raises(TypeError):   
            layer = nn.Conv2d(in_channels=3, out_channels=6, padding=0, kernel_size='kernel', stride=1, bias=True)         
            layer(input) 
 
    @timer
    def test_string_kernel_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        with pytest.raises(TypeError):
            layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding=0, kernel_size='size', stride=2)            
            layer(input)

    @timer
    def test_string_inp_channel_conv1d(self):
        input = torch.rand(1, 1, 10)
        with pytest.raises(TypeError): 
            layer = nn.Conv1d(in_channels='1', out_channels=1, padding=0, kernel_size=3, stride=1, bias=True)           
            layer(input)

    @timer
    def test_string_inp_channel_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        with pytest.raises(TypeError): 
            layer = nn.Conv2d(in_channels='input', out_channels=6, padding=0, kernel_size=3, stride=1, bias=True)
            layer(input) 

    @timer 
    def test_valid_string_inp_channel_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)       
        with pytest.raises(TypeError): 
            layer = torch.nn.Conv3d(in_channels='4', out_channels=1, padding=0, kernel_size=1, stride=2)           
            layer(input)

    @timer
    def test_valid_string_output_channel_conv1d(self):
        input = torch.rand(1, 1, 10)    
        with pytest.raises(TypeError):
            layer = nn.Conv1d(in_channels=1, out_channels='3', padding=0, kernel_size=3, stride=1, bias=True)            
            layer(input)

    @timer
    def test_valid_string_output_channel_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        with pytest.raises(TypeError):
            layer = nn.Conv2d(in_channels=3, out_channels='six', padding=0, kernel_size=3, stride=1, bias=True)            
            layer(input) 
 
    @timer
    def test_valid_string_output_channel_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        with pytest.raises(TypeError):
            layer = torch.nn.Conv3d(in_channels=1, out_channels='1', padding=0, kernel_size=1, stride=2)            
            layer(input)

    @timer
    def test_valid_dilation_conv1d(self):
        input = torch.rand(1, 1, 10)
        for dilation in [-100, -50, -1, 0]:
            with pytest.raises(RuntimeError):
                layer = nn.Conv1d(in_channels=1, out_channels=1, padding=0, kernel_size=3, stride=1, bias=True, dilation=dilation)            
                layer(input)

    @timer
    def test_valid_dilation_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        for dilation in [-50, -25, -1, 0]:
            with pytest.raises(RuntimeError):
                layer = nn.Conv2d(in_channels=3, out_channels=6, padding=0, kernel_size=20, stride=1, bias=True, dilation=dilation)            
                layer(input)

    @timer    
    def test_valid_dilation_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        for dilation in [-500, -5, -1, 0]:
            with pytest.raises(RuntimeError):            
                layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding=0, kernel_size=100, stride=2, dilation=dilation)
                layer(input)

    @timer
    def test_valid_kernel_conv1d(self):
        input = torch.rand(1, 1, 10)
        with pytest.raises(RuntimeError):
            layer = nn.Conv1d(in_channels=1, out_channels=1, padding=0, kernel_size=20, stride=1, bias=True)            
            layer(input)

    @timer
    def test_valid_kernel_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        with pytest.raises(RuntimeError):            
            layer = nn.Conv2d(in_channels=3, out_channels=6, padding=0, kernel_size=20, stride=1, bias=True)
            layer(input)   

    @timer
    def test_valid_kernel_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        with pytest.raises(RuntimeError):            
            layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding=0, kernel_size=100, stride=2)
            layer(input)

    @timer
    def test_string_padding_conv1d(self):
        input = torch.rand(1, 1, 10)
        with pytest.raises(TypeError):            
            layer = nn.Conv1d(in_channels=1, out_channels=1, padding='0', kernel_size=3, stride=1, bias=True)
            layer(input)

    @timer
    def test_string_padding_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        with pytest.raises(TypeError):            
            layer = nn.Conv2d(in_channels=3, out_channels=6, padding='abc', kernel_size=3, stride=1, bias=True)
            layer(input)   
    
    @timer
    def test_string_padding_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        with pytest.raises(TypeError):            
            layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding='1', kernel_size=3, stride=2)
            layer(input)
    
    @timer
    def test_valid_padding_conv1d(self):
        input = torch.rand(1, 1, 10)
        with pytest.raises(TypeError):  
            layer = nn.Conv1d(in_channels=1, out_channels=1, padding='same', kernel_size=5,  stride=1)        
            layer(input)    
    
    @timer
    def test_valid_padding_conv2d(self):
        input = torch.rand(1, 3, 4, 4)
        layer = nn.Conv2d(in_channels=3, out_channels=6, padding='xyz', kernel_size=20, stride=1, bias=True)
        with pytest.raises(TypeError):            
            layer(input)

    @timer    
    def test_valid_padding_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding='valid', kernel_size=1, stride=2)
        with pytest.raises(TypeError):            
            layer(input)      

    @timer
    def test_negative_or_zero_stride_conv1d(self):
        input = torch.rand(1, 1, 10)
        for stride in (0,-1,-2,-10):
            layer = nn.Conv1d(in_channels=1, out_channels=1, padding=0, kernel_size=5, stride=-1)
            with pytest.raises(RuntimeError):            
                layer(input)

    @timer
    def test_negative_or_zero_stride_conv2d(self):
        input = torch.randn(1, 3, 4, 4)
        for stride in (0,-1,-2,-10):
            layer = nn.Conv2d(in_channels=3, out_channels=6, padding=0, kernel_size=3, stride=stride, bias=True)
            with pytest.raises(RuntimeError):            
                layer(input)
    
    @timer
    def test_negative_or_zero_stride_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        for stride in (0,-1,-2, -10):
            layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding=0, kernel_size=3, stride=stride)
            with pytest.raises(RuntimeError):            
                layer(input)

    @timer    
    def test_dtype_conv1d(self):
        input = torch.rand(1, 1, 10)
        for stride in range(1, 100):
            layer = nn.Conv1d(in_channels=1, out_channels=1, padding=0, kernel_size=5, stride=1)
            assert layer(input).dtype==torch.float32

    @timer
    def test_dtype_conv2d(self):
        input = torch.randn(1, 3, 4, 4)
        for stride in range(1, 100):
            layer = nn.Conv2d(in_channels=3, out_channels=6, padding=0, kernel_size=3, stride=2, bias=True)
            assert layer(input).dtype==torch.float32

    @timer
    def test_dtype_conv3d(self):
        input = torch.randn(1, 1, 4, 4, 4)
        for stride in range(1, 100):           
            layer = torch.nn.Conv3d(in_channels=1, out_channels=1, padding=0, kernel_size=3, stride=stride)
            assert layer(input).dtype==torch.float32

class TestMaxPool:   

    @timer
    def test_negative_or_zero_stride_maxpool(self):
        input = torch.rand(1, 1, 3, 3)
        for stride in [-100, -50, -2]:                        
            with pytest.raises(RuntimeError): 
                max_pool = nn.MaxPool2d(kernel_size=3, stride=stride) 
                max_pool(input)

    @timer
    def test_string_stride_maxpool(self):
        input = torch.rand(1, 1, 3, 3)
        for stride in ['abc', '1', '100', '2']:
            with pytest.raises(TypeError):  
                max_pool = nn.MaxPool2d(kernel_size=3, stride=stride)                
                max_pool(input)

    @timer
    def test_negative_or_zero_kernel_size_maxpool(self):
        input = torch.rand(1, 1, 3, 3) 
        for kernel_size in [-100, -5, -1, 0]:
            with pytest.raises(RuntimeError): 
                max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1)               
                max_pool(input)

    @timer
    def test_string_kernel_size_maxpool(self):
        input = torch.rand(1, 1, 3, 3)
        for kernel_size in ['-100', 'abc', '1', '30']:
                with pytest.raises(TypeError): 
                    max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1)                     
                    max_pool(input)

    @timer
    def test_valid_dilation_maxpool(self):
        for dilation in ['-100', 'abc', '1', '30']:               
            input = torch.rand(1, 1, 3, 3)
            with pytest.raises(TypeError): 
                max_pool = nn.MaxPool2d(kernel_size=3, dilation=dilation)
                max_pool(input)

class TestSoftmax:

    @timer
    def test_string_dim_softmax(self):
        input = torch.randn(100, 100)
        for dim in ['0', '1', 'abc']:
            with pytest.raises(TypeError): 
                m = nn.Softmax(dim=dim)
                m(input)

    @timer
    def test_dim_range_softmax(self):
        input = torch.randn(100, 100)
        for dim in [-5, 4, 10, 100]:
            with pytest.raises(IndexError): 
                m = nn.Softmax(dim=dim)
                m(input)

    @timer
    def test_softmax_value(self):
        input = torch.zeros(4, 4)
        expected = torch.tensor([[0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25]])

        softmax_obj = Softmax(input, dim=1)
        assert torch.all(torch.eq(expected, softmax_obj.calculate_softmax()))
