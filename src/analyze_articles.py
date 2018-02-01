import os
import joblib as p
from article_loader import AData,get_date,post_json
import numpy as np
from WDM import WDM
import os
from time import sleep
from scoring import BM25


def load_articles():
    articles_dump = "../res/dataset.dat"
    if not os.path.isfile(articles_dump):
        dataset = AData()
        dataset.load_new()
        p.dump(dataset,open(articles_dump,"wb"))
    dataset = p.load(open(articles_dump,"rb"))
    print("Article dump loaded")
    return dataset

def load_dtf(dataset):
    dtm_dump = "../res/wdm.dat"
    if not os.path.isfile(dtm_dump):
        dm = WDM()
        dm.add_docs(dataset.get_latest(-1))
        p.dump(dm,open(dtm_dump,"wb"), protocol=0)
    dm = p.load(open(dtm_dump,"rb"))
    print("WDM dump loaded")
    return dm




def main():

    #load data
    dataset = load_articles()
    print(len(dataset.content),"\n\n\n")
    # load new articles
    # dataset.load_new()

    # load index
    dm = load_dtf(dataset)
    last_id = dm.get_last_doc_id()
    latest = dataset.get_latest(last_id)

    dm.add_docs(latest)
    dm.construct_idf()




        # fo = open("../res/articles/%d.txt"%i,"w")
        # fo.write(dataset.titles[i])
        # fo.write("\n")
        # fo.write(dataset.links[i])
        # fo.write("\n")
        # fo.write(dataset.content[i])
        # fo.write("\n")
    #load index
    # dm = load_dtf(dataset)
    # load new articles
    # dataset.load_new()
    # dm.add_doc(dataset.content)
    #init search engine
    search_bm25 = BM25(dm)



    for qq in range(len(dataset.content)):
        if dataset.links[qq].split("/")[2] in ['realnoevremya.ru','tatcenter.ru'] : continue
        art_range = dataset.get_last_two_days(qq)
        scores, ref = search_bm25.rank(dataset.titles[qq],art_range)
        print(dataset.titles[qq],dataset.links[qq].split("/")[2])
        show = min(10,len(scores))
        for i in range(show):
            article_id = dataset.ids.index(ref[i])
            if article_id != qq:
                if scores[i]/scores[0]<0.3: break
                print(scores[i]/scores[0]," ",dataset.titles[article_id])


def calculate_links(links):
    res = dict()
    for i in range(len(dataset.content)):
        lnk = dataset.links[i].split("/")[2]
        if lnk in res:
            res[lnk] += 1
        else:
            res[lnk] = 1

    for lnk in res:
        print(lnk,res[lnk])

if __name__ == "__main__":
    main()
