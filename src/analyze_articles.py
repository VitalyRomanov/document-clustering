import os
import joblib as p
from article_loader import AData, get_date, post_json
from WDM import WDM
import os
from scoring import BM25


index_by = 'titles'
enable_filtering = False
articles_dump = "../res/dataset.dat"
dtm_dump = "../res/wdm.dat"
similarity_threshold = .4


def load_articles(articles_path):
    if not os.path.isfile(articles_path):
        dataset = AData()
        dataset.load_new()
        dataset.save(articles_path)
        # save_articles(articles_path)
    # p.load(open(articles_path, "rb"))
    dataset = AData.load(articles_path)
    print("Article dump loaded")
    return dataset


# def save_articles(dataset, path):
#     p.dump(dataset, open(path, "wb"))


def load_dtf(dtm_path, dataset):
    if not os.path.isfile(dtm_path):
        dm = WDM()
        dm.add_docs(dataset.get_latest(-1, content_type=index_by, filter_bl=enable_filtering))
        dm.save(dtm_path)
        # save_dtf(dm, dtm_path)
    # dm = p.load(open(dtm_path, "rb"))
    dm = WDM.load(dtm_path)
    print("WDM dump loaded")
    return dm


# def save_dtf(dm, path):
#     p.dump(dm, open(path, "wb"), protocol=0)


def main():
    # load data
    dataset = load_articles(articles_dump)
    print(len(dataset.content), "\n\n\n")

    # load new articles
    dataset.load_new()

    # load index
    dm = load_dtf(dtm_dump, dataset)

    last_id = dm.get_last_doc_id()
    latest = dataset.get_latest(last_id, content_type=index_by, filter_bl=enable_filtering)
    dm.add_docs(latest)

    dataset.save(articles_dump)
    dm.save(dtm_dump)

    # init search engine
    search_bm25 = BM25(dm)

    titles = dataset.get_titles()

    for t_ord, title in enumerate(titles['titles']):
        scores, ref = search_bm25.rank(title)
        print(title)
        show = min(20, len(scores))
        for i in range(show):
            article_id = dataset.ids.index(ref[i])
            if titles["ids"][t_ord] != ref[i]:
                normalized_score = scores[i] / scores[0]
                if normalized_score < similarity_threshold:
                    break
                print(normalized_score, " ", dataset.titles[article_id])


def calculate_links(links):
    res = dict()
    for i in range(len(dataset.content)):
        lnk = dataset.links[i].split("/")[2]
        if lnk in res:
            res[lnk] += 1
        else:
            res[lnk] = 1

    for lnk in res:
        print(lnk, res[lnk])


if __name__ == "__main__":
    main()
