import json
import os
import pickle
from datetime import datetime
import urllib.request
import numpy as np

def date2int(date):
    return int(datetime.strptime(date,'%Y-%m-%d  %H:%M:%S').timestamp())

def get_date(ts):
    return datetime.fromtimestamp(
        int(repr(ts))
    ).strftime('%Y-%m-%d %H:%M:%S')

def load_latest():
    dump_file = "articles_dump.dat"
    l_time = 1509031277
    if os.path.isfile(dump_file):
        articles = p.load(open(dump_file,"rb"))
    else:
        articles = []
    return articles

def retreive_articles(l_time):
    data = json.load(open('1509031277.json'))
    # retreive articles' dates
    dates = list(map(date2int,map(lambda x: x['public_date'],data)))
    # sort articles by date
    s_ind = sorted(range(len(dates)), key=lambda k: dates[k])
    s_data = [data[ind] for ind in s_ind]
    return s_data

def retreive_articles_url(time):
    url_addr = "https://www.business-gazeta.ru/index/monitoring/timestamp/%d"%time
    data = None
    with urllib.request.urlopen(url_addr) as url:
        data = json.loads(url.read().decode())
        # print(data)
    dates = list(map(date2int,map(lambda x: x['public_date'],data)))
    # sort articles by date
    s_ind = sorted(range(len(dates)), key=lambda k: dates[k])
    s_data = [data[ind] for ind in s_ind]
    return s_data

def post_json(data_json):
    url_addr = "https://www.business-gazeta.ru/index/similar"
    enc_json = data_json.encode('utf-8')
    req = urllib.request.Request(url_addr, data=enc_json,
                            headers={'content-type': 'application/json'})
    response = urllib.request.urlopen(req)
    print(response.read())


def get_last_time(articles):
    return articles[-1] if len(articles)!=0 else 0
    latest = 0
    for article in articles:
        candidate = date2int(article['public_date'])
        if candidate > latest:
            latest = candidate
    return latest

def get_sections(s_data):
    # split data into sections
    ids = list(map(lambda x:x['id'],s_data))
    titles = list(map(lambda x:x['title'],s_data))
    content = list(map(lambda x:x['content'],s_data))
    dates = list(map(date2int,map(lambda x: x['public_date'],s_data)))
    links = list(map(lambda x:x['link'],s_data))
    return ids,titles,content,dates,links

class AData:
    ids = None
    titles = None
    content = None
    dates = None
    links = None
    _TWO_DAYS = 60*60*24*2 # sec*min*hr*2d

    def __init__(self):
        self.ids = []
        self.titles = []
        self.content = []
        self.dates = []
        self.links = []

        ids,titles,content,dates,links = get_sections(load_latest())
        self.join_sections(ids,titles,content,dates,links)
        self._latest = get_last_time(self.dates)
        self.new = len(self.ids)

    def load_new(self):
        self._latest = get_last_time(self.dates)
        self.new = len(self.ids)
        last_date = self.dates[-1] if len(self.dates)>0 else 1509031277
        print("Retreiving after %s"%get_date(last_date),end = ": ")
        new_articles = retreive_articles_url(last_date)
        ids,titles,content,dates,links = get_sections(new_articles)
        self.join_sections(ids,titles,content,dates,links)
        self.new = len(new_articles)
        if self.new == 0:
            print("Nothing new")
        else:
            print("%d added"%self.new)

    def join_sections(self,ids,titles,content,dates,links):
        self.ids+=ids
        self.titles+=titles
        self.content+=content
        self.dates+=dates
        self.links+=links

    def get_article(self,id):
        return self.content[id]

    def get_last_time(self):
        return self.dates[-1]

    def two_days_range(self,id1,id2):
        TWO_DAYS = 60*60*24*2
        return True if abs(self.dates[id1]-self.dates[id2])<TWO_DAYS else False

    def get_last_two_days(self,a_id):
        ids = []
        for i in range(a_id,-1,-1):
            if self.two_days_range(a_id,i):
                ids.append(i)
            else:
                break
        return np.array(ids)

    def make_json(self,doc_id,similar_id):
        return json.dumps({"article_id":self.ids[doc_id],
                    "similar_id":[self.ids[s_id] for s_id in similar_id]},
                    indent=4)

    def get_latest(self,last_id):
        """
        Input: last_id - the id in self.ids.
        Returns: all documents and ids that appear after the doc with last_id
        """
        black_list = ['realnoevremya.ru','tatcenter.ru']
        try:
            last_pos = self.ids.index(last_id)
        except:
            if last_id!=-1:
                raise Exception("No document with such id")
            last_pos = last_id

        latest_ids = []
        latest_content = []
        for i in range(last_pos+1,len(self.ids)):
            url = self.links[i].split("/")[2]
            if url not in black_list:
                latest_ids.append(self.ids[i])
                latest_content.append(self.titles[i])

        return {'ids':latest_ids,'docs':latest_content}
