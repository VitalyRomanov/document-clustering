from collections import Counter
from tokenization import get_document_words
import pickle
from pymystem3 import Mystem

def multimomial_lm(doc):

    return lm

class MLM:
    def __init__(self,doc):
        words = get_document_words(doc)
        self.lm = Counter(words)
        self.total_count = sum(self.lm.values())
        self.tc = self.total_count

    def getProb(self,token):
        return self.lm[token]/self.total_count+1e-300

    def getCount(self,token):
        return self.lm[token]

    def store(self,name):
        with open(name,"wb") as lm_file:
            pickle.dump(self,lm_file)

    def lemmatize(self):
        lm = Counter()
        m = Mystem()
        for token,count in self.lm.items():
            lemma = m.lemmatize(token)[0]
            if lemma in lm:
                lm[lemma] += count
            else:
                lm[lemma] = count
        self.lm = lm

class JMLM:
    def __init__(self,d_lm,r_lm,l):
        self.d_lm = d_lm
        self.r_lm = r_lm
        self.l = l

    def getProb(self,token):
        p_d = self.d_lm.getProb(token)
        p_r = self.r_lm.getProb(token)
        l = self.l
        return (1-l)*p_d+l*p_r

    def getCount(self,token):
        raise NotImplemented
