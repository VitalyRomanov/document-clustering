import pickle
from scipy.sparse import dok_matrix,csr_matrix,csc_matrix,lil_matrix
from scipy import multiply,dot
import numpy as np


c_w_ds = pickle.load(open("doc_freq_matr","rb"))
p_w_cs = pickle.load(open("ref_prob","rb"))
d_is  = np.matrix(pickle.load(open("doc_len","rb")))
print(c_w_ds.shape)
print(p_w_cs.shape)
print(d_is.shape)

g_mu_w = lambda w_id,mu: np.sum((c_w_ds[:,w_id].multiply((d_is - 1) * p_w_cs[0,w_id] - c_w_ds[:,w_id] + 1)).todense() / \
                            (( d_is - 1 + mu ) * ( c_w_ds[:,w_id].todense() - 1 + mu * p_w_cs[0,w_id])))

g_d_mu_w = lambda w_id,mu: np.sum(- (c_w_ds[:,w_id] * ( (d_is - 1) * p_w_cs[0,w_id] - c_w_ds[:,w_id] + 1)**2) / \
                                ((d_is - 1 + mu)**2 * (c_w_ds[:,w_id] - 1 + mu * p_w_cs[0,w_id])**2))


w_id = 1
mu = 800.
# print(c_w_ds[:,w_id].multiply(((d_is - 1) * p_w_cs[0,w_id] - c_w_ds[:,w_id] + 1)))
# print(c_w_ds[:,w_id].todense() - 1 + mu * p_w_cs[0,w_id])
# print(g_mu_w(w_id,mu))

print(( c_w_ds[:,w_id].todense() - 1 + mu * p_w_cs[0,w_id])*(( d_is - 1 + mu )))
# print(type(( d_is - 1 + mu )),type(( c_w_ds[:,w_id].todense() - 1 + mu * p_w_cs[0,w_id])))
