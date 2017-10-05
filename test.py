import pickle
from scipy.sparse import dok_matrix,csr_matrix,csc_matrix,lil_matrix,diags
from scipy import multiply,dot
import numpy as np
import tensorflow as tf




c_w_ds = pickle.load(open("doc_freq_matr","rb"))
p_w_cs = pickle.load(open("ref_prob","rb"))
d_is  = np.matrix(pickle.load(open("doc_len","rb")))
print(c_w_ds.shape)
print(p_w_cs.shape)
print(d_is.shape)

n_docs = d_is.shape[0]
voc_size = p_w_cs.shape[1]

g_mu_w = lambda w_id,mu: np.sum( c_w_ds[:,w_id].multiply( (d_is - 1) * p_w_cs[0,w_id] - c_w_ds[:,w_id] + 1 ) / \
                    np.multiply( d_is - 1 + mu , c_w_ds[:,w_id].todense() - 1 + mu * p_w_cs[0,w_id] ) )

g_d_mu_w = lambda w_id,mu: np.sum( - c_w_ds[:,w_id].multiply(np.square((d_is - 1) * p_w_cs[0,w_id] - c_w_ds[:,w_id] + 1)) / \
                    np.multiply( np.square(d_is - 1 + mu) , np.square(c_w_ds[:,w_id].todense() - 1 + mu * p_w_cs[0,w_id] ) ) )


w_id = 1
mu = 1292.27

# for i in range(1000):
#
#     g_mu = 0.; g_mu_d = 0.
#     for w_id in range(p_w_cs.shape[1]):
#         g_mu += g_mu_w(w_id,mu)
#         g_mu_d += g_d_mu_w(w_id,mu)
#
#     mu = mu - g_mu/g_mu_d
#     print("Iteration %d mu : %f"%(i,mu))

doc_len_norm = np.zeros(shape=(n_docs,))
for i in range(n_docs):
    if d_is[i] != 0:
        doc_len_norm[i] = 1/d_is[i]




d_i = diags(doc_len_norm,0)
p_w_ds = d_i*c_w_ds

p_d1 = tf.placeholder(shape=(1,voc_size),dtype=tf.float32)
p_d2 = tf.placeholder(shape=(1,voc_size),dtype=tf.float32)
l1 = tf.placeholder(dtype=tf.float32)
l2 = tf.placeholder(dtype=tf.float32)
p_ref = tf.placeholder(shape=(1,voc_size),dtype=tf.float32)

kl_dist = tf.reduce_sum(tf.multiply(p_d1,
                    tf.log((1-l1) * p_d1 + l1 * p_ref) - \
                    tf.log((1-l2) * p_d2 + l2 * p_ref)))

# def smoothed_kl_dist(p_d1,p_d2,p_ref,mu,d1,d2):
#     l1 = mu/(mu+d1)
#     l2 = mu/(mu+d2)
#     p_d1 = (p_d1 * (1-l1) + p_ref * l1).todense()
#     p_d2 = (p_d2 * (1-l2) + p_ref * l2).todense()
#     dist = 0
#     dist = np.sum( np.multiply(p_d1 , (np.log(p_d1) - np.log(p_d2))) )
#     return dist

print(type(p_w_cs))

print("Checking distanse")
with open("doc_dist_kl.txt","w") as d_dist:
 with tf.Session() as sess:
  for d1_id in range(n_docs):
   d_dist.write("%d "%d1_id)
   print("\r%d/%d"%(d1_id,n_docs),end = " ")
   for d2_id in range(n_docs):
    dist = sess.run(kl_dist,{p_d1:p_w_ds[d1_id,:].todense(),
                             p_d2:p_w_ds[d2_id,:].todense(),
                             l1:mu/(mu+d_is[d1_id,0]),
                             l2:mu/(mu+d_is[d2_id,0]),
                             p_ref:p_w_cs.todense()})
    d_dist.write("%f "%dist)
   d_dist.write("\n")






# def convert_sparse_matrix_to_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return tf.SparseTensor(indices, coo.data, coo.shape)
