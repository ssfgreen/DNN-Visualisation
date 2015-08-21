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
import csv
import datetime
from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
import base64

# Imports
from IDAPICourseworkLibrary import *
import h5py
import json
from pymongo_store import *

# inmport tsne plotting and created bn_saving tools
from tsne import bh_sne
# from bn_saving import *
import matplotlib.pyplot as plt

''' this file is very important!
    - it saves the weights, activations and parameters from the neural network
    - also plots the tsne graphs and saves those into the required folder structure too
'''

def save_weight_bias_slow(experiment_folder, filename, epoch, output_layer, ending, out_type):

    print("saving weights/biases...")
    
    # check for directory, if not create it
    # data_directory = check_create_directory("data/weights-biases")

    subfolder = experiment_folder + "/weights-biases"
    data_directory = check_create_directory(subfolder)

    # collecting all Tensor Shared Variables [W b W b W b] - weights and biases
    all_params = lasagne.layers.get_all_params(output_layer)
    # collecting a List of Numpy Arrays - all weights and biases
    param_values = lasagne.layers.get_all_param_values(output_layer)
    # another way to do this
    # all_param_values = [p.get_value() for p in all_params]

    # checking sizes
    no_arrays = len(all_params)
    print ("all params: ", all_params)
    print ("no arrays: ", no_arrays)

    # initialise layer number
    layer = 0

    # Going through, and saving each matrix to a separate file (not printing all, too slow??)
    for i, data in enumerate(param_values):
        
        w_or_b = "na"

        # odd ones are weights, indexed from one?
        if (i % 2) != 0:
            w_or_b = "bias"
            
        else:
            w_or_b = "weights"
            layer += 1

        # saving the filename: epoch, layer, w/b
        filename_unique = "{}/{}-E{}-L{}-{}.{}".format(data_directory, filename, epoch['number'], layer, w_or_b, ending)

        # print filename_unique
        print data.shape

        # different methods to output data - not all work :|
        if(out_type=="CSV"):
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif(out_type=="PICKLE"):
            with open(filename, 'w') as f:
                pickle.dump(data, f)
        elif(out_type=="NUMPY"):
            numpy.savetxt(filename_unique, data, delimiter=",")
        elif(out_type=="JSON"):
            print "yet to be implemented"

    print("weights/biases saved!") # hurray


def save_activations_test(experiment_folder, filename, epoch, dataset, output_layer, ending, out_type):

    print ("Saving Activations...")
    
    # check for directory, if not create it
    
    subfolder = experiment_folder + "/activations"
    data_directory = check_create_directory(subfolder)

    # collects all lasagne layers as Theano Variables
    th_layers = lasagne.layers.get_all_layers(output_layer)

    X_val = dataset['X_valid']
    # print X_val.eval()

    for i, layer in enumerate(th_layers):
        # only care about layers with params
        if not (layer.get_params() or isinstance(layer, lasagne.layers.FeaturePoolLayer)):
            continue

        data = lasagne.layers.get_output(layer, X_val, deterministic=True).eval()

        filename_unique = "{}/{}-E{}-L{}.{}".format(data_directory, filename, epoch['number'], i, ending)

        # print filename_unique
        print data.shape

        # different methods to output data - not all work :|
        if(out_type=="CSV"):
            with open(filename, "wb") as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif(out_type=="PICKLE"):
            with open(filename, 'w') as f:
                pickle.dump(data, f)
        elif(out_type=="NUMPY"):
            numpy.savetxt(filename_unique, data, delimiter=",")
        elif(out_type=="JSON"):
            print "yet to be implemented"
        
    print("Activations saved!")


def save_params (exID, experiment_folder, filename, output_layer, datafile, num_epochs, batch_size, num_hidden_units, learning_rate,
    momentum, epoch, dataset):

    train_loss = epoch['train_loss']
    valid_loss = epoch['valid_loss']
    valid_accuracy =  (epoch['valid_accuracy'] * 100)
    epoch_num = epoch['number']

    output_dim = dataset['output_dim']
    input_dim = dataset['input_dim']
    X_val = dataset['valid_set']

    print "X_val shape: ", X_val.shape

    print("Saving Parameters...")

    # subfolder = "data/parameters"

    subfolder = experiment_folder + "/params"
    data_directory = check_create_directory(subfolder)

    params_to_save = dict(
        # NETWORK_LAYERS = [str(type(layer)) for layer in lasagne.layers.get_all_layers(output_layer)],
        DATA_FILENAME = datafile,
        NUM_EPOCHS = num_epochs,
        BATCH_SIZE = batch_size,
        NUM_HIDDEN_UNITS = num_hidden_units,
        LEARNING_RATE = learning_rate,
        MOMENTUM = momentum,
        TRAIN_LOSS = train_loss,
        VALID_LOSS = valid_loss,
        VALID_ACCURACY = valid_accuracy,
        OUTPUT_DIM = output_dim,
        INPUT_DIM = input_dim,
        EPOCH = epoch_num
        )

    # open('file.json', 'w') as f: f.write(json.dumps(members))

    # log = json.dumps(params_to_save)
    filename = "params"
    filename = "{}/{}-E{}.json".format(data_directory, filename, epoch_num)

    # mongo stuff
    dbClient = DatabaseClient()

    filteredResults = dbClient.query(exID)

    name = "{}, Accuracy: {}".format(experiment_folder, valid_accuracy)

    if (filteredResults == []):
      experiment = {
        "TDID": exID,
        "HUMAN_NAME": name,
        "PARAMS": params_to_save,
        "DATA": {}
      }
      dbClient.insert(experiment)
    else:
      resultId = filteredResults[0]['_id']
      experiment = dbClient.get(resultId)
      experiment["PARAMS"] = params_to_save

      updatedObject = dbClient.update(resultId, experiment)
      print "updated: ", updatedObject

    # folder saving stuff
    with open(filename,"w") as outfile:
        outfile.write(json.dumps(params_to_save,sort_keys=True,
             indent=4, separators=(',', ': ')))

    print("Parameters Saved!")


def plot_activations(exID, exNUM, experiment_folder, epoch, dataset, output_layer, num):
    
    # coords_data_directory = check_create_directory("/sne_coords")

    # coords_subfolder = experiment_folder + "/sne_coords"
    # coords_data_directory = check_create_directory(coords_subfolder)

    plot_subfolder = experiment_folder + "/sne_plots"
    plot_data_directory = check_create_directory(plot_subfolder)

    print ("Calculating Activations...")
    
    # collects all lasagne layers as Theano Variables
    th_layers = lasagne.layers.get_all_layers(output_layer)

    X_val = dataset['X_valid']
    y = dataset['y_valid_raw']

    print "X_val shape: ", X_val.shape
    print "y_val shape: ", y.shape
    # print X_val.eval()

    for i, layer in enumerate(th_layers):
        # only care about layers with params
        if not (layer.get_params() or isinstance(layer, lasagne.layers.FeaturePoolLayer)):
            continue

        # gets the activations
        data = lasagne.layers.get_output(layer, X_val, deterministic=True).eval()

        # converts activations to x,y coords
        coords, labels = plot_bn_sne(data, y, num)

        # putting into list for saving to mongo
        coords_array = np.reshape(coords, (1,-1))
        coords_list = coords_array.tolist()[0]
        labels_array = np.reshape(labels, (1,-1))
        labels_list = labels_array.tolist()[0]

        print "coords list", coords_list
        # mongo stuff
        dbClient = DatabaseClient()

        filteredResults = dbClient.query(exID)

        if (filteredResults == []):
          experiment = {
            "TDID": exID,
            "DATA": {
              "LAYER": [i], 
              "EPOCH": [epoch['number']],
              "TSNE_DATA": [
                coords_list
              ],
              "TSNE_LABELS": labels_list
            }
          }
          dbClient.insert(experiment)
        else:
          resultId = filteredResults[0]['_id']
          experiment = dbClient.get(resultId)

          if "TSNE_DATA" in experiment["DATA"]:
            existing_coords = experiment["DATA"]["TSNE_DATA"]
            print "some lists"
            existing_coords.append(coords_list)
          else:
            print "no lists"
            existing_coords = [coords_list]
      
          if "TSNE_LABELS" in experiment["DATA"]:
            existing_labels = experiment["DATA"]["TSNE_LABELS"]
          else:
            existing_labels = labels_list

          if "EPOCH" in experiment["DATA"]:
            existing_epochs = experiment["DATA"]["EPOCH"]
            existing_epochs.append(epoch['number'])
          else:
            existing_epochs = [epoch['number']]

          if "LAYER" in experiment["DATA"]:
            existing_layer = experiment["DATA"]["LAYER"]
            existing_layer.append(i)
          else:
            existing_layer = [i]

          experiment["DATA"]["LAYER"] = existing_layer
          experiment["DATA"]["EPOCH"] = existing_epochs
          experiment["DATA"]["TSNE_DATA"] = existing_coords
          experiment["DATA"]["TSNE_LABELS"] = existing_labels

          updatedObject = dbClient.update(resultId, experiment)
          print "updated: ", updatedObject

        # generate appropriate file names
        # coords_filename_unique = "{}/C-{}-EX{}-E{}-L{}.csv".format(coords_data_directory, exID, exNUM, epoch['number'], i)
        # labels_filename_unique = "{}/L-{}.csv".format(coords_data_directory, exID)
        plot_filename_unique = "{}/{}-E{}-L{}".format(plot_data_directory, "sne_plot", epoch['number'], i)
        
        # save stuff
        # numpy.savetxt(coords_filename_unique, coords, delimiter=",")
        # numpy.savetxt(labels_filename_unique, labels, delimiter=",")

        # plot and save
        plt.scatter(coords[:, 0], coords[:, 1], c=labels)
        plt.savefig(plot_filename_unique, dpi=120)
        plt.close()






##### HELPER FUNCTIONS #####

def plot_bn_sne(data, labels, size):

  print "data[0]: ", data.shape
  print "labels[0]: ", labels.shape

  # trim the data & labels down to reasonable size
  data = data[0:size]
  labels = labels[0:size]

  # sizes
  data0 = data.shape[0]
  data1 = data.shape[1]

  # dimensionality reduction with bn_sne
  X_2d = bh_sne(data)
  print "plot shape: ", X_2d.shape

  return X_2d, labels
  # plot & save
  plot_save(filename, X_2d, labels, data0, data1)

def time_date_id():
  date_today = time.strftime('%d%m%Y')
  time_now = time.strftime('%H%M%S')

  return "{}{}".format(date_today, time_now)

def new_experiment_folder(subfolder):

  # append date to subfolder
  date_today = time.strftime('%d-%b-%Y') 
  time_now = time.strftime('%H-%M-%S')
  subfolder = subfolder + "/" + date_today

  # check if that folder exists, if not create it
  check_create_directory(subfolder)

  # get next experiment number
  num = new_dir_index(subfolder)

  # new folder name
  foldername = "{}/ex-{}".format(subfolder,num)

  # check if that folder exists, if not create it
  check_create_directory(foldername)

  print "fn: ", foldername
  return foldername, num


def check_create_directory(new_folder):

    # finds the current path, and creates new data directory
    path = os.path.dirname(os.path.realpath(__file__))
    data_directory = path + "/" + new_folder

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    print "dd: ", data_directory
    return data_directory

def new_dir_index(sub_folder):

  # get list of all directories
      # finds the current path, and creates new data directory
  path = os.path.dirname(os.path.realpath(__file__))
  data_directory = path + "/" + sub_folder

  all_dirs = [name for name in os.listdir(data_directory) ] # if os.path.isdir(name)

  # split each directory "ex-N" to get highest N
  last_experiment = 0
  print len(all_dirs)

  if len(all_dirs) > 0:

    print ">0"
    for i in range(len(all_dirs)):
      # assuming folder is 'ex-N'
      front, dash, end = all_dirs[i].rpartition("-")
      if len(front) > 0:
        end = int(end)
        if end > last_experiment:
          last_experiment = end

    # return new experiment number
    return last_experiment + 1

  # otherwise 
  return 1

def unused_suff():

  # finds the current directory path
  path = os.path.dirname(os.path.realpath(__file__))

  # gets todays date and time
  date_today = time.strftime('%d-%b-%Y')
  time_now = time.strftime('%H-%M-%S')

  # creates a file directory, checks if it exists, if not creates one
  data_directory = "{}/images/{}".format(path, date_today)
  if not os.path.exists(data_directory):
      os.makedirs(data_directory)

  # create the filename
  filename = 'images/{}/bh_sne-0x{}-1x{}-t{}.png'.format(date_today, Xdim0, Xdim1, time_now)


if __name__ == '__main__':

  print time_date_id()