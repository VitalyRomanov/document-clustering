# import MySQLdb
import sys
from getpass import getpass
import os
import pickle



def get_data():
 '''
 Provides the dictionary of articles
 Dictionary key : id of the article as it is stored in the DB
 The article content is stored as a dictionary. The content is
     title       : the title of the article
     body        : the body of the article
     timestamp   : the date of publication
 '''
 passw = getpass(prompt="Password for db: ")
 db = MySQLdb.connect(user      =   "root",
                      passwd    =   passw,
                      db        =   "articles_db",
                      charset   =   'utf8')
 c = db.cursor()

 articles = {}

 id_counter = 0
 c.execute("""SELECT id,title,article FROM external_articles""")
 result = c.fetchone()
 while result is not None:
  id_counter += 1
  a_id = result[0]
  a_title = result[1]
  a_body = result[2]
  articles[id_counter] = {'title':a_title,'body':a_body,'timestamp':0}
  result = c.fetchone()
 return articles


def get_all_articles():
    if os.path.isfile("articles.dat"):
        print("Loading articles from local dump")
        articles = pickle.load(open("articles.dat","rb"))
    else:
        print("Loading articles from db")
        articles = get_data()
        pickle.dump(open("articles.dat","wb"),articles)
