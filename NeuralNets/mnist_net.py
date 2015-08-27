from __future__ import print_function

# as lasagne is pre-release, some warnings pop up: now ignoring these!
import warnings
warnings.filterwarnings('ignore', module='lasagne')

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
# sys.path.insert(0, '../Helpers')

# this finds the files in other directories
sys.path.insert(0, './Helpers/database')
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

class Net:

    def __init__(self, DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
        NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG):
            self.DATA_URL = DATA_URL
            self.DATA_FILENAME = DATA_FILENAME
            self.NUM_EPOCHS = NUM_EPOCHS
            self.BATCH_SIZE = BATCH_SIZE
            self.NUM_HIDDEN_UNITS = NUM_HIDDEN_UNITS
            self.LEARNING_RATE = LEARNING_RATE
            self.MOMENTUM = MOMENTUM
            self.DEBUG = DEBUG

    # def check(self, data_url = None):
    #     if data_url is None:
    #         print("none")
    #         data_url = self.DATA_URL
    #     print (data_url)


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
        """Get data with labels, split into training, validation and test set."""
        data = self._load_data()
        X_train, y_train = data[0]
        X_valid, y_valid = data[1]
        X_test, y_test = data[2]

        if(self.DEBUG):
            print ("X_train shape: ", X_train.shape)
            print ("y_train shape: ", y_train.shape)
            print ("X_valid shape: ", X_valid.shape)
            print ("y_valid shape: ", y_valid.shape)
            print ("X_test shape: ", X_test.shape)
            print ("y_test shape: ", y_test.shape)

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
            input_dim=X_train.shape[1],
            output_dim=10,
        )


    def build_model(self, input_dim, output_dim,
                    batch_size=None, num_hidden_units=None):
        """Create a symbolic representation of a neural network with `intput_dim`
        input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
        layer.

        The training function of this model must have a mini-batch size of
        `batch_size`.

        A theano expression which represents such a network is returned.
        """

        if batch_size is None:
            batch_size = self.BATCH_SIZE
        if num_hidden_units is None:
            num_hidden_units = self.NUM_HIDDEN_UNITS

        # INPUT: 600 * 784
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, input_dim),
        )

        # LAYER 1: 512, ReLU
        l_hidden1 = lasagne.layers.DenseLayer(
            l_in,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
        l_hidden1_dropout = lasagne.layers.DropoutLayer(
            l_hidden1,
            p=0.5,
        )

        # LAYER 2: 512, ReLU
        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1_dropout,
            num_units=num_hidden_units,
            nonlinearity=lasagne.nonlinearities.rectify,
        )
        l_hidden2_dropout = lasagne.layers.DropoutLayer(
            l_hidden2,
            p=0.5,
        )

        # OUTPUT: 10 classes, softmax 
        l_out = lasagne.layers.DenseLayer(
            l_hidden2_dropout,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax,
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
        
        if num_epochs is None:
            num_epochs = self.NUM_EPOCHS

        dataset = self.load_data()

        print("Building model and compiling functions...")
        output_layer = self.build_model(
            input_dim=dataset['input_dim'],
            output_dim=dataset['output_dim'],
        )
        iter_funcs = self.create_iter_functions(dataset, output_layer)

        print("Starting training...")

        # creating an experiment folder for file structure
        experiment_folder, experiment_num = new_experiment_folder("experiments")
        exID = time_date_id()

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

                if epoch['number'] % 2 == 0:
                # if epoch['number'] == 1:
                    num_coords = 500
                    plot_activations(exID, experiment_num, experiment_folder, epoch, dataset, output_layer, num_coords)
                if epoch['number'] % 10 == 0:
                    save_activations_test(experiment_folder, "testing", epoch, dataset, output_layer, "csv", "NUMPY")
                    save_weight_bias_slow(experiment_folder, "testing", epoch, output_layer, "csv", "NUMPY")

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


def run_net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
    NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG):
        Net_running = Net(DATA_URL, DATA_FILENAME, NUM_EPOCHS, BATCH_SIZE, 
    NUM_HIDDEN_UNITS, LEARNING_RATE, MOMENTUM, DEBUG)
        Net_running.main()


if __name__ == '__main__':
        
    run_net('http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz', 
        500, 600, 512, 0.01, 0.9, False)
    # Net1.main()
