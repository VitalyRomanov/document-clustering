import MySQLdb
import sys

def get_data():
 db = MySQLdb.connect(user="root",passwd="d74paj2c",db="articles_db",charset='utf8')
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
  articles[id_counter] = {'title':a_title,'body':a_body}
  result = c.fetchone()
 return articles

# counter = 0
# with open(os.getcwd()+"/"+"samples.txt","w") as of:
#  for a_id,content in articles.items():
#   counter += 1
#   of.write("%s\n%s\n\n\n"%(content['title'],content['body']))
#   print(get_document_words(content['body']))
#   if counter>2: break
