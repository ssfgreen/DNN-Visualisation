 
import numpy as np
import gzip, cPickle
import matplotlib
matplotlib.use('TkAgg') # used to avoid attribute error from moviepy
import matplotlib.pyplot as plt
import time
import os

# used for getting all files in a folder
import pandas as pd
import sys
import glob

# importing the BarnesHut SNE (fast tSNE)
from tsne import bh_sne
import colorsys # enables going from hsl to rgb

# pca and lda imports
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy



def append_labels():
    my_data = np.genfromtxt('../data/CSV/testing-E20-L3.csv', delimiter=',')
    my_labels = np.genfromtxt('../data/CSV/labels-E136-L3.csv', delimiter=',')

    print "data shape", my_data.shape
    print "labels shape", my_labels.shape

    data = my_data[:500]
    labels = my_labels[:500]

    print data.shape
    print labels.shape

    labels = np.reshape(labels, (500,-1))

    # data = np.reshape(data, (-1,500))
    # labels = np.reshape(labels, (-1,500))

    new_data = np.append(labels, data, axis=1)

    print new_data.shape

    new_data = np.reshape(new_data, (500,-1))

    print new_data.shape
  
    np.savetxt("../data/CSV/lables.csv", new_data, delimiter=",")


if (__name__ == "__main__"):
  append_labels()