from data_reader import get_data
from language_model import MLM
import pickle


import os
from tokenization import get_document_words

def merge_articles(art):
 c = ""
 for v in art.values():
  c += v['body'] + "\n"
 return c

# Configure working directory
current_folder = os.getcwd()
lm_folder = current_folder + "/" + "models"
# Retreive data from db on local
articles = get_data()

lm_c = MLM(merge_articles(articles))
lm_c.store(lm_folder + "/" + "lm_c.lm")


lm_articles = {}
for a_id,content in articles.items():
 lm_articles[a_id] = MLM(content['body'])
 # lm = MLM(content['body'])
 # lm.store(lm_folder + "/" + "lm_%d.lm"%a_id)
 print(a_id)

with open(lm_folder + "/" + "lm_art.lmc","wb") as lm_art:
 pickle.dump(lm_articles,lm_art)


# counter = 0
# with open(os.getcwd()+"/"+"samples.txt","w") as of:
#  for a_id,content in articles.items():
#   counter += 1
#   of.write("%s\n%s\n\n\n"%(content['title'],content['body']))
#   print(get_document_words(content['body']))
#   if counter>2: break
