import json
from pprint import pprint
import os
import glob
import sys

def getIdFromFile():
  # gets files at that path
  path = os.path.dirname(os.path.realpath(__file__))
  folder = path + "/data"

  # glob.glob alows therminal stype collecting of data, i.e the *.csv
  allFiles = glob.glob(folder + "/*.json")

  # gets all the file names, removing full path with 'basename' and removing extension with splitext()[0]
  names = [os.path.splitext(os.path.basename(x))[0] for (x) in allFiles]

  IDs = []
  
  for name in names: 
    IDs.append(name)

  return IDs

def getJSONdata(filename):
  # gets files at that path
  path = os.path.dirname(os.path.realpath(__file__))
  fileAct = path + "/data/" + filename

  with open(fileAct) as data_file:    
    data = json.load(data_file)

  return data

if __name__ == '__main__':

  print "------- IDS -------"
  allIDs = getIdFromFile()
  pprint(allIDs)

  print "------- DATA --------"
  data = getJSONdata('271015175012.json')
  pprint(data)
