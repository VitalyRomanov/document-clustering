from data_reader import get_data
from language_model import MLM,mlm_optimal_parameter,mlm_optimal_parameter_tf
import pickle


import os
from tokenization import get_document_words

def merge_articles(art):
 c = ""
 for v in art.values():
  c += v['body'] + "\n"
 return c


current_folder = os.getcwd()
lm_folder = current_folder + "/" + "models"

articles = get_data()

print("Collection LM")
lm_c = MLM(merge_articles(articles))
# lm_c.store(lm_folder + "/" + "lm_c.lm")``

print("Documents' LM")
lm_articles = {}
for a_id,content in articles.items():
 lm_articles[a_id] = MLM(content['body'])

# with open(lm_folder + "/" + "lm_art.lmc","wb") as lm_art:
#  pickle.dump(lm_articles,lm_art)

print("Calculating optimal parameter")
mu = mlm_optimal_parameter_tf(lm_articles,lm_c)

print("Optimal : ",mu)


# counter = 0
# with open(os.getcwd()+"/"+"samples.txt","w") as of:
#  for a_id,content in articles.items():
#   counter += 1
#   of.write("%s\n%s\n\n\n"%(content['title'],content['body']))
#   print(get_document_words(content['body']))
#   if counter>2: break
