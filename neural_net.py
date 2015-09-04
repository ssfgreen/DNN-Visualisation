import sys
import math
sys.path.insert(0, './NeuralNets')

from mnist_net import *
from mnist_conv import *


class NeuralNet:

    # this is called when we do a post request
    def run(self,data):

        # this bit should be what runs
        print('running with {}'.format(data))

        # and returns this to...?
        return {'ID': '1239383828'}

    # this is called when we do a results get request
    def result(self,timedate):

        # some middle stuff
        print('accessing data with timedate {}'.format(timedate))

        # this is the data passed back 
        return {'data':'hello'}


def run_neural_net():
    NUM_EPOCHS = 5
    net1 = Net()
    # say_hello()

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def power_range(start, end, divisor):
    while start <= end:
        yield start
        start += start/float(divisor)

def run_neural_networks():
    
    last_value = -1;

    # change the number of units
    for x in power_range(1,2300, 1.7):
        units = int(x)
        if units != last_value:
            print units
            run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz',50, 600, units, 0.01, 0.9, False)
            last_value = units

    # change the momentum
    for x in power_range(0.1,1.0,1.7):
        m = x
        if m != last_value:
            print m
            run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz',30, 600, 800, 0.01, m, False)
            last_value = m


def run_conv_nets():
    last_value = -1;

    for x in power_range(1,160, 1.5):
        filters = int(x)
        if filters != last_value:
            print filters
            run_conv_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz', 20, 10000, 500, 0.01, 0.9, filters, 5, 2, False)
            last_value = filters


def run_meta_vis():
    print 'hello'


if __name__ == '__main__':
    run_neural_networks()
    run_conv_nets()
    # ---- RUN NET - ARGS!! ----
    # run_net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
    # NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG):

    # ---- RUN CONV NET - ARGS!! ----
    # run_conv_net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
    #         NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, NUM_FILTERS, FILTER_SIZE,
    #         POOL_SIZE, DEBUG)