from collections import Counter
from tokenization import get_document_words
import pickle
from pymystem3 import Mystem
import numpy as np
import tensorflow as tf
import sys
import scipy


'''
implements several language models that include smoothing:
    MLM : Multinomial language model
    JMLM : Language model with Jelinek Mercer smoothing
    DLM : Language model with Dirichlet smoothing
reference : https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf
'''

LOW_VALUE = sys.float_info.min

class MLM:
    '''
    Classical language model
    Counts number of word occurences in a diven document
    doc : document text used for generating language model
    '''
    def __init__(self,doc):
        words = get_document_words(doc)
        self.lm = Counter(words)
        self.total_count = sum(self.lm.values())
        self.tc = self.total_count

    def getProb(self,token):
        # returns probability of a token
        if self.tc != 0:
            return self.lm[token]/self.total_count+LOW_VALUE
        else:
            # in case the document is empty
            return LOW_VALUE

    def getCount(self,token):
        return self.lm[token]

    def store(self,name):
        # write the LM on disk
        with open(name,"wb") as lm_file:
            pickle.dump(self,lm_file)


class JMLM:
    '''
    Language model with Jelinek Mercer smoothing
    d_lm : document language model
    r_lm : reference language model, usually LM of the collection
    l : smoothing constant
    '''
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

    def getVoc(self):
        return self.d_lm.lm.keys()

class DMLM:
    '''
    Language model with Dirichlet smoothing
    d_lm : document language model
    r_lm : reference language model, usually LM of the collection
    mu : smoothing constant
    '''
    def __init__(self,d_lm,r_lm,mu):
        self.d_lm = d_lm
        self.r_lm = r_lm
        self.mu = mu

    def getProb(self,token):
        l = mu / (self.d_lm.tc + mu)
        return (1 - l) * self.d_lm.getProb(token) + l * self.r_lm.getProb(token)

    def getCount(self,token):
        raise self.d_lm.getCount(token)

    def getVoc(self):
        return self.d_lm.lm.keys()



def dlm_optimal_parameter(lm_docs,lm_c):
    '''
    Find the optimal parameter for language model with Dirichlet smoothing
    lm_docs : dictionaty of language models with doc_id as key
    lm_c : language model of a colection
    '''
    mu = 1200.

    d_size = len(lm_docs)   # number of documents
    v_size = len(lm_c)      # vocabulary size

    d_is = np.zeros([d_size,1], dtype=np.float32) # document sizes (in tokens)
    c_w_ds = scipy.dok_matrix((d_size,v_size), dtype=np.float32)   # token count matrix
    p_w_cs = np.zeros([v_size,0], dtype=np.float32)

    for w_i,w in enumerate(lm_c.getVoc()):
        p_w_cs[w_i,0] = lm_c.getProb(w);
        for lm_i,lm_doc in enumerate(lm_docs.values()):
            d_is[lm_i,0] = lm_doc.tc
            c_w_ds[lm_i,w_i] = lm_doc.getCount(w)



    for i in range(1000):

        g_mu_w = lambda w_id: np.sum((c_w_ds[:,w_id] * ((d_is - 1) * p_w_cs[w_id,0] - c_w_ds[:,w_id] + 1)) / \
                                    (( d_is - 1 + mu ) * ( c_w_ds[:,w_id] - 1 + mu * p_w_cs[w_id,0])))

        g_d_mu_w = lambda w_id: np.sum(- (c_w_ds[:,w_id] * ( (d_is - 1) * p_w_cs[w_id,0] - c_w_ds[:,w_id] + 1)**2) / \
                                        ((d_is - 1 + mu)**2 * (c_w_ds[:,w_id] - 1 + mu * p_w_cs[w_id,0])**2))

        g_mu = sum(map(g_mu_w,range(len(lm_c.getVoc()))))
        g_d_mu = sum(map(g_d_mu_w,range(len(lm_c.getVoc()))))

        mu = mu - g_mu / g_mu_d
        print("Iteration %d : %f"%(i,mu))

    # d_is = np.zeros([v_size,1], dtype=np.float32)
    # # p_w_cs = np.zeros([v_size,1], dtype=np.float32)
    # c_w_ds = np.zeros([v_size,1], dtype=np.float32)
    #
    # for i in range(1000):
    #     g_mu = 0.; g_mu_d = 0.
    #     for w_i,w in enumerate(lm_c.lm.keys()):
    #         p_w_cs = lm_c.getProb(w); pos = 0
    #         for lm_doc in lm_docs.values():
    #             c_w_da = lm_doc.lm[w]; d_ia = lm_doc.tc
    #             d_is[pos,0] = d_ia;# p_w_cs[pos,1] = p_w_c;
    #             c_w_ds[pos,0] = c_w_da
    #             pos += 1
    #         g_mu += np.sum((c_w_ds * ((d_is - 1) * p_w_cs - c_w_ds + 1)) / (( d_is - 1 + mu ) * ( c_w_ds - 1 + mu * p_w_cs)))
    #         g_mu_d += np.sum(- (c_w_ds * ( (d_is - 1) * p_w_cs - c_w_ds + 1)**2) / ((d_is - 1 + mu)**2 * (c_w_ds - 1 + mu * p_w_cs)**2))
    #         # print("\r %d/%d   "%(w_i,len(lm_c.lm)), end='')
    #     # print("")
    #     mu = mu - g_mu / g_mu_d
    #     print("Iteration %d : %f"%(i,mu))



    # for i in range(1000):
    #     g_mu = 0.; g_mu_d = 0.
    #     for w_i,w in enumerate(lm_c.lm.keys()):
    #         p_w_c = lm_c.getProb(w)
    #         for lm_doc in lm_docs.values():
    #             c_w_d = lm_doc.lm[w]
    #             d_i = lm_doc.tc
    #             # print((( d_i - 1 + mu ) * ( c_w_d - 1 + mu * p_w_c)))
    #             # print(d_i," ", mu," ", c_w_d," ", p_w_c)
    #             g_mu += (c_w_d * ((d_i - 1) * p_w_c - c_w_d + 1)) / \
    #                     (( d_i - 1 + mu ) * ( c_w_d - 1 + mu * p_w_c))
    #             g_mu_d += - (c_w_d * ( (d_i - 1) * p_w_c - c_w_d + 1)**2) / \
    #                         ((d_i - 1 + mu)**2 * (c_w_d - 1 + mu * p_w_c)**2)
    #         print("\r %d/%d   "%(w_i,len(lm_c.lm)), end='')
    #     print("")
    #     mu = mu - g_mu / g_mu_d
    #     print("Iteration %d : %f"%(i,mu))

    return mu

def mlm_optimal_parameter_tf(lm_docs,lm_c):
    v_size = len(lm_docs)

    with tf.device("/gpu:0"):
        d_is = tf.placeholder(tf.float32,shape=(v_size,1),name="doc_len")
        c_w_ds = tf.placeholder(tf.float32,shape=(v_size,1),name="doc_freq_count")
        p_w_cs = tf.placeholder(tf.float32,name="word_prob")
        # p_w_cs = tf.Variable(0,0,dtype=tf.float32,name="word_prob")

        mu = tf.Variable(1.01,dtype=tf.float32,name="mu")
        g = tf.Variable(0.,dtype=tf.float32,name="g_mu")
        g_d = tf.Variable(0.,dtype=tf.float32,name="g_mu_d")

        g_mu_num = tf.multiply(c_w_ds,((d_is - 1)*p_w_cs - c_w_ds + 1))
        g_mu_den = tf.multiply( d_is - 1 + mu, c_w_ds - 1 + p_w_cs*mu)
        g_mu = tf.reduce_sum(tf.div(g_mu_num,g_mu_den))

        g_mu_d_num = -tf.multiply(c_w_ds,tf.square((d_is - 1)*p_w_cs - c_w_ds + 1))
        g_mu_d_den = tf.multiply( tf.square(d_is - 1 + mu), tf.square(c_w_ds - 1 + p_w_cs*mu))
        g_mu_d = tf.reduce_sum(tf.div(g_mu_d_num,g_mu_d_den))

        g_ass = tf.assign(g,g+g_mu)
        g_d_ass = tf.assign(g_d,g_d+g_mu_d)
        mu_ass = tf.assign(mu,mu-g_ass/g_d_ass)

        init = tf.global_variables_initializer()

        d_i = np.zeros([v_size,1],dtype=np.float32)
        c_w_d = np.zeros([v_size,1],dtype=np.float32)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./graphs",sess.graph)
        print("Start computation")
        sess.run(init)
        mu_final = 0.
        for i in range(1000):
            for w_i,w in enumerate(lm_c.lm.keys()):
                p_w_c = lm_c.getProb(w); pos = 0
                for lm_doc in lm_docs.values():
                    c_w_da = lm_doc.lm[w]; d_ia = lm_doc.tc
                    d_i[pos,0] = d_ia;# p_w_cs[pos,1] = p_w_c;
                    c_w_d[pos,0] = c_w_da
                    pos += 1
                sess.run([g_ass,g_d_ass],{d_is:d_i,c_w_ds:c_w_d,p_w_cs:p_w_c})
                print("\r %d/%d   "%(w_i,len(lm_c.lm)), end='')
            print("")
            tf.assign(mu,mu-g/g_d)
            mu_final = sess.run(mu_ass)
            print("Iteration %d : "%i,mu_final)
    return mu_final
