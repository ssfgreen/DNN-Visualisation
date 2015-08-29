from __future__ import print_function


import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
import math

# from mnist import _load_data
# from mnist import create_iter_functions
# from mnist import train

sys.path.insert(0, '../helpers/database') # NOTE the ./ -> ../ when running from neural_net.py
from neural_net_saving import *
from tsne import bh_sne

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)




class ConvNet:

    def __init__(self, DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
        NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, NUM_FILTERS, FILTER_SIZE,
        POOL_SIZE, DEBUG):

        self.DATA_URL = DATA_URL
        self.DATA_FILENAME = DATA_FILENAME
        self.NUM_EPOCHS = NUM_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_HIDDEN_UNITS = NUM_HIDDEN_UNITS
        self.LEARNING_RATE = LEARNING_RATE
        self.MOMENTUM = MOMENTUM
        self.DEBUG = DEBUG
        self.NUM_FILTERS = NUM_FILTERS
        self.FILTER_SIZE = FILTER_SIZE
        self.POOL_SIZE = POOL_SIZE

    def _load_data(self, url=None, filename=None):
        """Load data from `url` and store the result in `filename`."""
        if url is None:
            url = self.DATA_URL
        if filename is None: 
            filename = self.DATA_FILENAME

        if not os.path.exists(filename):
            print("Downloading MNIST dataset")
            urlretrieve(url, filename)

        with gzip.open(filename, 'rb') as f:
            return pickle_load(f, encoding='latin-1')

    def load_data(self):
        data = self._load_data()
        X_train, y_train = data[0]
        X_valid, y_valid = data[1]
        X_test, y_test = data[2]

        # reshape for convolutions
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
        X_valid = X_valid.reshape((X_valid.shape[0], 1, 28, 28))
        X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))

        return dict(
            X_train=theano.shared(lasagne.utils.floatX(X_train)),
            y_train=T.cast(theano.shared(y_train), 'int32'),
            X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
            y_valid=T.cast(theano.shared(y_valid), 'int32'),
            valid_set = X_valid,
            y_valid_raw = y_valid,
            X_test=theano.shared(lasagne.utils.floatX(X_test)),
            y_test=T.cast(theano.shared(y_test), 'int32'),
            num_examples_train=X_train.shape[0],
            num_examples_valid=X_valid.shape[0],
            num_examples_test=X_test.shape[0],
            input_height=X_train.shape[2],
            input_width=X_train.shape[3],
            input_dim=[X_train.shape[2],X_train.shape[3]],
            output_dim=10,
            )


    def build_model(self, input_width, input_height, output_dim,
                    batch_size=None):

        if batch_size is None:
            batch_size = self.BATCH_SIZE

        num_hidden_units = self.NUM_HIDDEN_UNITS
        no_filters = self.NUM_FILTERS # 32
        double_filters = no_filters*2
        filter_size = self.FILTER_SIZE # 5,5
        pool_size = self.POOL_SIZE # 2,2
        

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, 1, input_width, input_height),
            )

        l_conv1 = lasagne.layers.Conv2DLayer(
            l_in,
            num_filters=no_filters,
            filter_size=(filter_size, filter_size),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
            )
        l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(pool_size, pool_size))

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=double_filters,
            filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
            )
        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(pool_size, pool_size))

        l_hidden1 = lasagne.layers.DenseLayer(
            l_pool2,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(),
            )

        l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

        l_out = lasagne.layers.DenseLayer(
            l_hidden1_dropout,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.GlorotUniform(),
            )

        return l_out

    def create_iter_functions(self, dataset, output_layer,
                              X_tensor_type=T.matrix,
                              batch_size=None,
                              learning_rate=None, momentum=None):
        """Create functions for training, validation and testing to iterate one
           epoch.
        """

        if batch_size is None:
            batch_size = self.BATCH_SIZE
        if learning_rate is None:
            learning_rate = self.LEARNING_RATE
        if momentum is None:
            momentum = self.MOMENTUM

        batch_index = T.iscalar('batch_index')
        X_batch = X_tensor_type('x')
        y_batch = T.ivector('y')
        batch_slice = slice(batch_index * batch_size,
                            (batch_index + 1) * batch_size)

        output = lasagne.layers.get_output(output_layer, X_batch)
        loss_train = lasagne.objectives.categorical_crossentropy(output, y_batch)
        loss_train = loss_train.mean()

        output_test = lasagne.layers.get_output(output_layer, X_batch,
                                                deterministic=True)

        loss_eval = lasagne.objectives.categorical_crossentropy(output_test,
                                                                y_batch)

        loss_eval = loss_eval.mean()

        pred = T.argmax(output_test, axis=1)
        accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

        all_params = lasagne.layers.get_all_params(output_layer)

        updates = lasagne.updates.nesterov_momentum(
            loss_train, all_params, learning_rate, momentum)

        iter_train = theano.function(
            [batch_index],
            loss_train,
            updates=updates,
            givens={
                X_batch: dataset['X_train'][batch_slice],
                y_batch: dataset['y_train'][batch_slice],
            },
        )

        iter_valid = theano.function(
            [batch_index],
            [loss_eval, accuracy],
            givens={
                X_batch: dataset['X_valid'][batch_slice],
                y_batch: dataset['y_valid'][batch_slice],
            },
        )

        iter_test = theano.function(
            [batch_index],
            [loss_eval, accuracy],
            givens={
                X_batch: dataset['X_test'][batch_slice],
                y_batch: dataset['y_test'][batch_slice],
            },
        )



        return dict(
            train=iter_train,
            valid=iter_valid,
            test=iter_test,
        )

    def train(self, iter_funcs, dataset, batch_size=None):
        """Train the model with `dataset` with mini-batch training. Each
           mini-batch has `batch_size` recordings.
        """

        if batch_size is None:
            batch_size = self.BATCH_SIZE

        num_batches_train = dataset['num_examples_train'] // batch_size   # // means floor division
        num_batches_valid = dataset['num_examples_valid'] // batch_size

        for epoch in itertools.count(1): # counts from 1
            batch_train_losses = []
            for b in range(num_batches_train):
                batch_train_loss = iter_funcs['train'](b)
                batch_train_losses.append(batch_train_loss)

            avg_train_loss = np.mean(batch_train_losses)

            batch_valid_losses = []
            batch_valid_accuracies = []
            for b in range(num_batches_valid):
                batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
                batch_valid_losses.append(batch_valid_loss)
                batch_valid_accuracies.append(batch_valid_accuracy)

            avg_valid_loss = np.mean(batch_valid_losses)
            avg_valid_accuracy = np.mean(batch_valid_accuracies)

            yield {
                'number': epoch,
                'train_loss': avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
            }

    def main(self,num_epochs=None):

        print("Loading data...")
        dataset = self.load_data()

        if num_epochs is None:
            num_epochs = self.NUM_EPOCHS
            
        print("Building model and compiling functions...")
        output_layer = self.build_model(
            input_height=dataset['input_height'],
            input_width=dataset['input_width'],
            output_dim=dataset['output_dim'],
            )

        iter_funcs = self.create_iter_functions(
            dataset,
            output_layer,
            X_tensor_type=T.tensor4,
            )


        experiment_folder, experiment_num = new_experiment_folder("experiments")
        exID = time_date_id()

        print("Starting training...")
        now = time.time()
        try:
            for epoch in self.train(iter_funcs, dataset):
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch['number'], num_epochs, time.time() - now))
                now = time.time()
                print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
                print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
                print("  validation accuracy:\t\t{:.2f} %%".format(
                    epoch['valid_accuracy'] * 100))

                if epoch['number'] == 1:
                    save_params(exID, experiment_folder, "testing", output_layer, self.DATA_FILENAME, self.NUM_EPOCHS, self.BATCH_SIZE, self.NUM_HIDDEN_UNITS, self.LEARNING_RATE, 
                        self.MOMENTUM, epoch, dataset)

                if epoch['number'] % 1 == 0:
                # if epoch['number'] == 1:
                    num_coords = 500
                    plot_activations(exID, experiment_num, experiment_folder, epoch, dataset, output_layer, num_coords)
                if epoch['number'] % 10 == 0:
                    print("not saving yet")
                    # save_activations_test(experiment_folder, "testing", epoch, dataset, output_layer, "csv", "NUMPY")
                    # save_weight_bias_slow(experiment_folder, "testing", epoch, output_layer, "csv", "NUMPY")

                if epoch['number'] >= num_epochs:
                    # save_params(output_layer, datafile, num_epochs, batch_size, num_hidden_units, learning_rate
                 # momentum, train_loss, valid_loss, valid_accuracy, output_dim, input_dim)
                    save_params(exID, experiment_folder, "testing", output_layer, self.DATA_FILENAME, 
                        self.NUM_EPOCHS, self.BATCH_SIZE, self.NUM_HIDDEN_UNITS, self.LEARNING_RATE, 
                        self.MOMENTUM, epoch, dataset)

                    # after the experiment has concluded, run the meta and pca graph plotting
                    meta_pca_sne(exID,experiment_folder)
                    tsne_pca(exID,experiment_folder)
                    break

        except KeyboardInterrupt:
            pass

        return output_layer


def run_conv_net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
        NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, NUM_FILTERS, FILTER_SIZE,
        POOL_SIZE, DEBUG):
        Net_running = ConvNet(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
        NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, NUM_FILTERS, FILTER_SIZE,
        POOL_SIZE, DEBUG)
        Net_running.main()

if __name__ == '__main__':
    run_conv_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz',
        30, 10000, 500, 0.01, 0.9, 32, 5, 2, False)


# def run_net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
#     NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG):
#         Net_running = Net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
#     NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG)
#         Net_running.main()


# if __name__ == '__main__':
        
#     run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz', 
#         500, 600, 512, 0.01, 0.9, False)

#     def __init__(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
#         NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, NUM_FILTERS, FILTER_SIZE,
#         POOL_SIZE, DEBUG):

# NUM_EPOCHS = 500
# BATCH_SIZE = 600
# LEARNING_RATE = 0.01
# MOMENTUM = 0.9