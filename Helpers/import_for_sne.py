import numpy as np
import matplotlib.pyplot as plt
import base64
import pandas as pd
import os
import sys
import glob
import math
import re

# new_data = np.genfromtxt('test_csv.csv', dtype=np.float32, delimiter=',')


def find_files(extension):
  path = os.path.dirname(os.path.realpath(__file__))
  folder = path + extension

  # glob.glob alows therminal stype collecting of data, i.e the *.csv
  allFiles = glob.glob(folder + "/*.csv")

  # print glob.glob(path)

  print folder

  return allFiles

  # big_array = [] #  empty regular list
  #   for i in range(5):
  #       arr = i*np.ones((2,4)) # for instance
  #       big_array.append(arr)
  #   big_np_array = np.array(big_array)  # transformed to a numpy array

def method_one_concatenation():

  allFiles = find_files("/csv_colah")

  # initiates by getting first file
  intitial = np.genfromtxt(allFiles[0], delimiter=",")
  # trims the NaNs from the end (the '\n's)
  intitial = intitial[:,:-1]
  # concatenates all rows 
  intitial = np.reshape(intitial, (1,-1))

  count = 0

  # goes through all the files in the list of files saved
  for i, file in enumerate(allFiles):
    # to not append to the one we initialised with above 
    if i != 0:
      # gets the contents
      df = np.genfromtxt(file, delimiter=",")
      # trims the NaN
      df = df[:,:-1]
      # concatenates all rows
      df = np.reshape(df, (1,-1))
      # appends to first one (need to all be the same shape)
      intitial = np.append(intitial, df, axis=0)
      count = count  + 1
      print count

  
  # trim to make reasonable
  # intitial = intitial[0:500]

  print "shape: ", intitial.shape
  np.savetxt("test.csv", intitial, delimiter=",")

def method_two_nested():

  allFiles = find_files("/csv_colah")

  # initiates by getting the first file
  intitial = np.genfromtxt(allFiles[0], delimiter=",")
  # trims the NaNs from the end (the '\n's)
  intitial = intitial[:,:-1]

  count = 0

  # goes through all the files in the list of files saved
  for i, file in enumerate(allFiles):
    # to not append to the one we initialised with above 
    if i != 0:
      # gets the contents
      df = np.genfromtxt(file, delimiter=",")
      # trims the NaN
      df = df[:,:-1]
      # appends to first one (need to all be the same shape)
      intitial = np.append(intitial, df, axis=0)
      count = count  + 1
      print count

  print "shape: ", intitial.shape

  np.savetxt("test2.csv", intitial, delimiter=",")

# The Pandas method found online
def pandas():

  path =r"C:\DRO\DCL_rawdata_files"
  allFiles = glob.glob(path + "/*.csv")
  frame = pd.DataFrame()
  list = []
  for file in allFiles:
      df = pd.read_csv(str.join(path,file),index_col=None, header=0)
      list.append(df)
  frame = pd.concat(list)

def save_file_names_indexed(allFiles):
  # # gets files at that path
  # path = os.path.dirname(os.path.realpath(__file__))
  # folder = path + "/colah_indexed"

  # # glob.glob alows therminal stype collecting of data, i.e the *.csv
  # allFiles = glob.glob(folder + "/*.csv")

  # gets all the file names, removing full path with 'basename' and removing extension with splitext()[0]
  names = [os.path.splitext(os.path.basename(x))[0] for (x) in allFiles]

  labels = []

  for name in names:
    labels.append(math.floor(float(name)))

  # print labels

  np.savetxt("test_labels.csv", np.asarray(labels), delimiter=",")

def save_labels_epoch_layer(str_date,str_ex):
  # gets files at that path
  path = os.path.dirname(os.path.realpath(__file__))
  folder = path + "/experiments/" + str_date + "/" + str_ex + "/sne_coords"

  # glob.glob alows therminal stype collecting of data, i.e the *.csv
  allFiles = glob.glob(folder + "/*.csv")

  # gets all the file names, removing full path with 'basename' and removing extension with splitext()[0]
  names = [os.path.splitext(os.path.basename(x))[0] for (x) in allFiles]

  epochs = []
  layers = []

  for name in names: 
    e = re.search('-E(.+?)-L', name)
    if e:
      epochs.append(float(e.group(1)))
    l = re.search('-L(.+?)',name)
    if l:
      layers.append(float(l.group(1)))

  layer_fn = 'layer-{}-{}-labels.csv'.format(str_date, str_ex)
  epoch_fn = 'epoch-{}-{}-labels.csv'.format(str_date, str_ex)

  np.savetxt(layer_fn, np.asarray(layers), delimiter=",")
  np.savetxt(epoch_fn, np.asarray(epochs), delimiter=",")

def method_concatenation_and_index(ext_path):

  allFiles = find_files(ext_path)

  # initiates by getting first file
  intitial = np.genfromtxt(allFiles[0], delimiter=",")

  # gets the index of files in colah files
  # save_file_names_indexed(allFiles)



  # trims the NaNs from the end (the '\n's)
  intitial = intitial[:,:-1]
  # concatenates all rows 
  intitial = np.reshape(intitial, (1,-1))

  count = 0

  # goes through all the files in the list of files saved
  for i, file in enumerate(allFiles):
    # to not append to the one we initialised with above 
    if i != 0:
      # gets the contents
      df = np.genfromtxt(file, delimiter=",")
      # trims the NaN
      df = df[:,:-1]
      # concatenates all rows
      df = np.reshape(df, (1,-1))
      # appends to first one (need to all be the same shape)
      intitial = np.append(intitial, df, axis=0)
      count = count  + 1
      print count

  print "shape: ", intitial.shape
  np.savetxt("test_data.csv", intitial, delimiter=",")


if __name__ == "__main__":
  # method_one_concatenation(); # shape: 60, 20000
  # method_two_nested();      # shape: 600000, 2
  # get_file_names()
  # method_concatenation_and_index("/colah_indexed")
  # allFiles = find_files("/csv_indexed")
  # save_labels_epoch_layer()
  str_date = "09-Aug-2015"
  str_ex = "ex-12"
  save_labels_epoch_layer(str_date,str_ex)