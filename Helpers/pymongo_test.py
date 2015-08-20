from pymongo import MongoClient
import datetime

'''
    this is the mongobd (through pymongo) example file for saving
    TODO - implement the saving 

'''


# this is the default: but can also call simply MongoClient()
client = MongoClient('mongodb://localhost:27017/') 

# to access the database
db = client['test_database']

# get the collection (can also use dot-notation)
collection = db['test-collection']

# in pymongo, we use dictionaries to represent the documents
post = {
  "author": "Mike",
  "text": "My first blog post!", 
  "tags": ["mongodb", "python", "pymongo"],
  "date": datetime.datetime.utcnow()
  }

# the insert_one method inserts a document into a collection
posts = db.posts
post_id = posts.insert_one(post).inserted_id

# the 'find_one' looks for a document containing "author":"Mike"
print posts.find_one({"author": "Mike"})

# adding more 
new_posts = [{
  "author": "Mike",
  "text": "Another post!",
  "tags": ["bulk", "insert"],
  "date": datetime.datetime(2009, 11, 12, 11, 14)
  },
  {
  "author": "Eliot",
  "title": "MongoDB is fun",
  "text": "and pretty easy too!",
  "date": datetime.datetime(2009, 11, 10, 10, 45)
  }]

result = posts.insert_many(new_posts)

print result.inserted_ids

# finds all the posts in the collection?
for post in posts.find():
  print post

# the same, but filters with author = mike
for post in posts.find({"author": "Mike"}):
  print post

# to count the number of posts of a certain kind (i.e - filter)
print posts.find({"author": "Mike"}).count()