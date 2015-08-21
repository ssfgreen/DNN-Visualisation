import numpy as np
import matplotlib.pyplot as plt
import base64
import pandas as pd
import os
import sys
import glob
import math
import re
import fnmatch
import json

def find_files(extension, filetype):
  path = os.path.dirname(os.path.realpath(__file__))
  folder = path + extension

  # glob.glob alows therminal stype collecting of data, i.e the *.csv
  allFiles = glob.glob(folder + filetype)

  return allFiles

def getIdFromFile():
  # gets files at that path
  path = os.path.dirname(os.path.realpath(__file__))
  folder = path + "/data"

  find_files("/Helpers/experiments", )

  # glob.glob alows therminal stype collecting of data, i.e the *.csv
  allFiles = glob.glob(folder + "/*.json")

  # gets all the file names, removing full path with 'basename' and removing extension with splitext()[0]
  names = [os.path.splitext(os.path.basename(x))[0] for (x) in allFiles]

  IDs = []
  
  for name in names: 
    IDs.append(name)

  return IDs

def test_walk():

  matches = []

  # for root, dirnames, filenames in os.walk('./Helpers/experiments'):
  #   for name in filenames:
  #       # finds the preface type, and gets string
  #       n = re.search('(.+?)-E', name)
  #       if n:
  #         n = n.group(1)
  #       # checks if param
  #       if (n == 'params'):
  #         print "params: ", name

  #         # gets the date and experiment
  #         r = re.search('ments/(.+?)/params', root)
  #         if r: 
  #           r = r.group(1)
  #         # replaces / to get an ID
  #         ID = r.replace("/","-")
  #         # generates the full files url
  #         url = os.path.join(root, name)
  #         print "params id: ", ID
  #         # opens the json file
  #         with open(url) as data_file:    
  #           data = json.load(data_file)
  #         print data
  #       if (n == 'coords'):
  #         print "coords: ", name
  #       if (n == 'labels'):
  #         print 'labels: ', name

  for root, dirnames, filenames in os.walk('./Helpers/experiments'):
    for name in dirnames:
      n = re.search('ex-(\d+)', name)
      if n: # i.e - if an exercise directory
        n = n.group(1)
        print "name: ", name, " ", n
        # get root date
        r = re.search('ments/(.+?$)', root)
        if r: 
          r = r.group(1)
          print "date: ", r
        ex_url = os.path.join(root, )


        # r = re.search('ments/(.+?)')
        # print root
        # print name
        # print(os.path.join(root, name))
  
  # for name in names: 
  #   e = re.search('-E(.+?)-L', name)
  #   if e:
  #     epochs.append(float(e.group(1)))
  #   l = re.search('-L(.+?)',name)
  #   if l:
  #     layers.append(float(l.group(1)))


  #  (
  #     PARAMS: (
  #     ID: 271015175012,
  #     BATCH_SIZE: 800,
  #     MOMENTUM: 0.7,
  #     NUM_EPOCHS: 200,
  #     NUM_HIDDEN_UNITS: 216
  #   ),
  #   DATA: (
  #     COORDS: [1,2,3,4,5,6],
  #     LAYER: [3,1,2],
  #     EPOCH: [30,40,10],
  #     TSNE_DATA: [ 
  #         [2,2,4,24,5,23,1,14,4,3],
  #         [7,7,10,10,14,14,20,20,40,40],
  #         [26,26,90,90,14,14,24,100,90,10]
  #     ],
  #     TSNE_LABELS: [2,6,3,4,3]
  #   )
  # )

  # (
  #   ID: 01919393838123123,
  #   url: ./JSON/dafds.json
  #   human_name: "20-AUG-15, ex-1"
  #  )

  # generate unique ID based upon file path
  # find the params json
  # create human readable name based upon file path and params
  # create new dictonary 
  # add params from the imported json to dict['PARAMS']
  # find the sne_coords folder
  # for each file
    # split the filename into layer and epoch
    # import the values, append the xy's (reshape)
    # append the layer and epoch to the dictonrary 'DATA'
    # append the list of xy values to the data array
  # save the dictionary in a ____.json that uses the unique ID in the json folder
  # save the ID, the above filename as the url, and the human readable name into another directory
    # that stores the indexing jsons

  # when place a query, read all the files in there, add them to a dictionary and return
  # as a json




def dictoriary():
  meta_data = {}
  meta_data['ID'] = {}
  meta_data['ID']['PARAMS'] = {}
  meta_data['ID']['PARAMS']['ID'] = 271015175012

  print meta_data

if __name__ == '__main__':
  # dictoriary()
  test_walk()
