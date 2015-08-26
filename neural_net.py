import sys
import math
sys.path.insert(0, './NeuralNets')

from mnist_net import *


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

    for x in power_range(1, 510, 2):
        epoch = int(x)
        if epoch != last_value:
            print epoch
            run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz',30, 600, epoch, 0.01, 0.9, False)
            run_meta_vis()
            last_value = epoch

def run_meta_vis():
    print 'hello'


if __name__ == '__main__':

    print "running"
    # ---- RUN NET - ARGS!! ----
    # run_net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
    # NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG):

    # run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz', 
    # 500, 600, 512, 0.01, 0.9, False)

    # run_neural_networks()
    # run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz',40, 600, 512, 0.01, 0.9, False)