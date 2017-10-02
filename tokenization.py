import re

def get_document_words(document):
    # Simple tokenization without normalization
    tokens = list(filter(lambda x: x!= '', re.split(r"[^A-Za-z0-9А-Яа-я]", document.lower())))
    return tokens

def chunk_text(doc,chunking):
    if chunking == 'words':
        chunks = get_document_words(doc)
    elif chunking == 'lines':
        chunks = doc.split("\n")
    else:
        raise NotImplemented
    return chunks
