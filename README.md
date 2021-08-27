It contains different tests for Convolution, Maxpool and Softmax nodes. Also, This has a 3node graph implemented which runs for 10 times taking random inputs everytime and generating the results on the basis of that input.

## Prerequisites
One needs to have torch 1.2.0

## Steps to run the tests
pytest -v

## To run the test and capture the timings of each test function
pyest -sv

## To run the 3node graph
python three_node_graph.py