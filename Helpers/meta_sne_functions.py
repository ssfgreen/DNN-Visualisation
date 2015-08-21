from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg') # used to avoid attribute error from moviepy
import matplotlib.pyplot as plt
import math
from tsne import bh_sne
# used in varima
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
from pymongo_store import *


def castPCA2(array):
    pca = PCA(n_components = array.shape[1])
    transform = pca.fit_transform(array)
    return transform


def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)
    

def meta_pca_sne(exID):
    # mongo stuff
    dbClient = DatabaseClient()

    filteredResults = dbClient.query(exID)

    if filteredResults is None:
      print "No results"
      return

    filteredId = filteredResults[0]['_id']
    experiment = dbClient.get(filteredId)

    list_of_coords = experiment['DATA']['TSNE_DATA']

    np_list = np.asarray(list_of_coords)
    print np_list.shape

    labels = np.asarray([1,2,3,4,5,6])

    sne_co = bh_sne(np_list, perplexity=1.0, theta=0.5)
    plt.scatter(sne_co[:,0], sne_co[:,1], c=labels)
    plt.show()

    flat_coords = np.reshape(sne_co, (1,-1))
    flat_coords = flat_coords.tolist()[0]

    experiment['DATA']['META'] = flat_coords

    updatedObject = dbClient.update(filteredId, experiment)
    print "updated: ", updatedObject


def tsne_pca(exID):

    # mongo stuff
    dbClient = DatabaseClient()

    filteredResults = dbClient.query(exID)

    if filteredResults is None:
      print "No results"
      return

    filteredId = filteredResults[0]['_id']
    experiment = dbClient.get(filteredId)

    list_of_coords = experiment['DATA']['TSNE_DATA']

    pca_list = []
    for coords in list_of_coords:
        np_val = np.asarray(coords)
        coords_array = np.reshape(coords, (-1,2))

        cast = castPCA2(coords_array)
        print "cast: ", cast.shape

        cast_veri = varimax(cast)
        print "cast_veri", cast_veri.shape
        pca_list.append(cast_veri)

    np_pca = np.asarray(pca_list)

    print "pca: ", np_pca.shape
    np_pca = np.reshape(np_pca, (6,-1))
    print "pca: ", np_pca.shape

    labels = np.asarray([1,2,3,4,5,6])

    sne_pca = bh_sne(np_pca, perplexity=1.0, theta=0.5)
    plt.scatter(sne_pca[:,0], sne_pca[:,1], c=labels)
    plt.show()

    flat_coords = np.reshape(sne_pca, (1,-1))
    flat_coords = flat_coords.tolist()[0]

    experiment['DATA']['PCA'] = flat_coords

    updatedObject = dbClient.update(filteredId, experiment)
    print "updated: ", updatedObject



if __name__ == '__main__':

  meta_pca_sne()
  tsne_pca()