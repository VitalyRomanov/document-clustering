import re
from pymystem3 import Mystem

def get_document_words(document):
    # m = Mystem()
    tokens = list(filter(lambda x: x!= '', re.split(r"[^A-Za-z0-9А-Яа-я]", document.lower())))
    # for t_i,token in enumerate(tokens):
    #     token = m.lemmatize(token)
    # tokens = Mystem().lemmatize(" ".join(tokens))
    return tokens

def chunk_text(doc,chunking):
    if chunking == 'words':
        chunks = get_document_words(doc)
    elif chunking == 'lines':
        chunks = doc.split("\n")
    else:
        raise NotImplemented
    return chunks
