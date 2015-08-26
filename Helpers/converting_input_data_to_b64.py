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
import base64
# sys.path.insert(0, '../Helpers')


PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)


DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'

def _load_data(url=None, filename=None):
        """Load data from `url` and store the result in `filename`."""
        if url is None:
            url = DATA_URL
        if filename is None: 
            filename = DATA_FILENAME

        if not os.path.exists(filename):
            print("Downloading MNIST dataset")
            urlretrieve(url, filename)

        with gzip.open(filename, 'rb') as f:
            return pickle_load(f, encoding='latin-1')


def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    return X_valid

def create_b64_string():
    X_val = load_data()
    small_X_val = X_val[:2000]
    valid_data_array = np.reshape(small_X_val, (1,-1))
    valid_data_list = valid_data_array.tolist()[0]

    print "type", type(valid_data_array[0][0])

    # convert to base64 as otherwise it's too large!
    new_data = base64.encodestring(valid_data_array[0])

    with open("save_b64", "wb") as text_file:
      text_file.write(new_data)


if __name__ == '__main__':
    create_b64_string()