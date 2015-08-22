import numpy as np
import gzip, cPickle
import matplotlib
matplotlib.use('TkAgg') # used to avoid attribute error from moviepy
import matplotlib.pyplot as plt
import time
import os
import itertools

# used for getting all files in a folder
import pandas as pd
import sys
import glob

# mongodb stuff
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId

from pymongo_store import *

# importing the BarnesHut SNE (fast tSNE)
from tsne import bh_sne
import colorsys # enables going from hsl to rgb

# pca and lda imports
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
# filtered = dbClient.query(12)
# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

def check_create_directory(new_folder):

    # finds the current path, and creates new data directory
    path = os.path.dirname(os.path.realpath(__file__))
    data_directory = path + "/" + new_folder

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # print "dd: ", data_directory
    return data_directory

def read_and_vid():

    dbClient = DatabaseClient()

    results = dbClient.query()

    for result in results:
    #   resultId = result['_id']

        # result = results[0]
        resultId = result['_id']

        data_object = dbClient.get(resultId)

        # for key, value in data_object["DATA"].iteritems():
        #     print key

        tsne_coords = data_object['DATA']['TSNE_DATA']
        tsne_labels = data_object['DATA']['TSNE_LABELS']
        exID = data_object['TDID']

        length = len(tsne_coords)
        no_points = len(tsne_coords[0])/2

        NPcoords = np.asarray(tsne_coords[0])
        NPcoords = np.reshape(NPcoords, (-1,2))

        i = 0
        for co in tsne_coords:
            if i != 0:
                coord = np.asarray(co)
                coord = np.reshape(coord, (-1,2))
                NPcoords = np.append(NPcoords, coord, axis=0)
            i = i+1

        print NPcoords.shape

        labels = np.asarray(tsne_labels)
        labels = np.reshape(labels, (500,-1))
        # coords = np.asarray(tsne_coords)
        # labels = np.asarray(tsne_labels)

        # print "coords shape", coords.shape
        # print "labels shape", labels.shape

        # coords = np.reshape(coords, (-1,500))
        filename = check_create_directory('experiments/gifs')
        filename = "{}/{}.gif".format(filename, exID)

        makeVideo(NPcoords, labels, no_points, length, filename)

def makeVideo(X_2d,labels,diff, samples,name):

    # name = "./experiments/video"
    # doesn't work atm: gives error (AttributeError: 'FigureCanvasMac' object has no attribute ....
    # name = "test4.gif"
    fps = 5
    duration = samples # the number of samples in the array

    print "X_2d shape", X_2d.shape

    def make_frame(t):
        fig = plt.figure()
        print(t)

        # trim the number of points
        X_ = X_2d[t*diff:t*diff+diff,0]
        # X = X_[:1000] # trim down
        Y_ = X_2d[t*diff:t*diff+diff,1]
        # Y = Y_[:1000] # trim down

        fig, ax = plt.subplots()
        ax.scatter(X_,Y_,c=labels)

        fig.patch.set_visible(False)
        ax.axis('off')

        return mplfig_to_npimage(fig)

    animation = mpy.VideoClip(make_frame, duration = duration)
    animation.write_gif(name, fps=fps)


if __name__ == '__main__':

    read_and_vid()