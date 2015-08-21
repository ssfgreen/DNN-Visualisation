from pymongo import MongoClient, ReturnDocument
from bson.objectid import ObjectId
import datetime

class DatabaseClient:

  # /testingNN is one that started to train
  # /testSet is the original testset

  def __init__(self, uri='mongodb://localhost:27017/testingNN'):
    client = MongoClient(uri)
    db = client.get_default_database()
    self.experiments = db['experiments']

  def get(self, _id):
    """ Return a single experiment object
    """ 

    experiment = self.experiments.find_one({'_id':ObjectId(_id)})

    if (experiment is None):
      raise LookupError("Experiment with id {} cant be found".format(_id))

    experiment['_id'] = str(experiment['_id'])

    return experiment

  def insert(self, experiment):
    """ insert an experiment into the database collection experiments, 
    """

    _id = self.experiments.insert_one(experiment).inserted_id

    return self.get(_id)


  def update(self, _id, experiment):
    """ Update the given experiment
    """

    if ('_id' in experiment):
      experiment.pop('_id')

    experiment = self.experiments.find_one_and_update(
      {'_id' : ObjectId(_id)},
      {'$set': experiment },
      return_document = ReturnDocument.AFTER
      )

    experiment['_id'] = str(experiment['_id'])
    return experiment

  def query(self, Cfilter=None):
    """ Returns all experiments 
    """ 

    results = []

    if Cfilter is None:
      for experiment in self.experiments.find():
        experiment['_id'] = str(experiment['_id'])
        results.append(experiment)
    else:
      collectionFilter = {}
      collectionFilter['TDID'] = Cfilter

      for experiment in self.experiments.find(collectionFilter):
        experiment['_id'] = str(experiment['_id'])
        results.append(experiment)
    
    return results

  def remove(self, _id):
    """ Remove a given experiment from the database
    """

    self.experiments.remove({'_id':ObjectId(_id)})

def test():

  dbClient = DatabaseClient()


  experiment = {
      "ID": 260115175012,
      "HUMAN_NAME": "26-JAN-1992, ex-00",
      "PARAMS": {
        "BATCH_SIZE": 600,
        "MOMENTUM": 0.9,
        "NUM_EPOCHS": 500,
        "NUM_HIDDEN_UNITS": 512
      },
      "DATA": {
        "COORDS": [1,1,2,2,10,2],
        "LAYER": [1,2,6],
        "EPOCH": [2,150,90],
        "TSNE_DATA": [
            [1,1,2,2,3,3,4,4,5,5], 
            [7,7,8,8,9,9,10,11,12,13],
            [12,12,14,24,15,23,12,14,14,23]
        ],
        "TSNE_LABELS": [2,6,3,4,4]
      }
    }

  experiment1 = {
    "ID": 271015175012,
    "HUMAN_NAME": "27-AUG-2015, ex-1",
    "PARAMS": {
      "BATCH_SIZE": 800,
      "MOMENTUM": 0.7,
      "NUM_EPOCHS": 200,
      "NUM_HIDDEN_UNITS": 216
    },
    "DATA": {
      "COORDS": [1,2,3,4,5,6],
      "LAYER": [3,1,2],
      "EPOCH": [30,40,10],
      "TSNE_DATA": [ 
          [2,2,4,24,5,23,1,14,4,3],
          [7,7,10,10,14,14,20,20,40,40],
          [26,26,90,90,14,14,24,100,90,10]
      ],
      "TSNE_LABELS": [2,6,3,4,3]
    }
  }

  experiment2 = {
      "TDID": 291015175012,
      "HUMAN_NAME": "20-AUG-2015, ex-12",
      "PARAMS": {
        "BATCH_SIZE": 200,
        "MOMENTUM": 0.4,
        "NUM_EPOCHS": 400,
        "NUM_HIDDEN_UNITS": 116
      },
      "DATA": {
        "COORDS": [2,2,5,4,9,12],
        "LAYER": [3, 1, 5],
        "EPOCH": [20,50,10],
        "TSNE_DATA": [
            [2,1,4,2,5,3,1,14,40,3],
            [7,1,10,2,14,4,20,23,40,41],
            [26,1,90,2,14,2,24,10,90,10]
        ],
        "TSNE_LABELS": [1,5,3,2,3]
      }
    }

  # dbClient.insert(experiment)
  # dbClient.insert(experiment1)
  # dbClient.insert(experiment2)

  results = dbClient.query()

  print "results: ", results

  # filtered = dbClient.query(12)

  # filteredId = filtered[0]['_id']
  # data_object = dbClient.get(filteredId)

  # if "TSNE_COORDS" in data_object["DATA"]:
  #   coord_list = data_object["DATA"]["TSNE_COORDS"]
  #   print "some lists"
  #   coord_list.append([2,3,4])
  # else:
  #   print "no lists"
  #   coord_list = [[1,2,3]]

  # data_object["DATA"]["TSNE_COORDS"] = coord_list
  # updatedObject = dbClient.update(filteredId, data_object)

  # print "updated Obj: ", updatedObject


  # print "filtered: ", filtered
  for result in results:
    resultId = result['_id']
    dbClient.remove(resultId)

  # new_results = dbClient.query()

  # print "new_results: ", new_results

  
  # firstId = results[0]['_id']


  # firstObject = dbClient.get(firstId)

  # print firstObject

  # firstObject['DATA']['LAYER'].append(8)
  # firstObject['DATA']['EPOCH'] = [3]

  # print firstObject

  # updatedId = dbClient.update(firstId, firstObject)

  # updatedObject = dbClient.get(firstId) 

  # print "updated", updatedObject

  return 


if (__name__ == "__main__"):
  test()