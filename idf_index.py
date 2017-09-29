from tokenization import get_document_words

def build_idf_index(vocabulary,collection):
    idf = {}
    for w in vocabulary:
        idf[w] = 0.
        for doc in collection:
            tokens = get_document_words(doc)
            if w in tokens:
                idf[w] += 1.
        idf[w] = log(len(collection)/idf[w])
    return idf
