import sys
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

if __name__ == '__main__':

    # ---- RUN NET - ARGS!! ----
    # run_net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
    # NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG):

    # run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz', 
    # 500, 600, 512, 0.01, 0.9, False)

    run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz', 
    20, 600, 1, 0.01, 0.9, False)
