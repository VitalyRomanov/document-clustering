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

p_w_ds = csr_matrix(c_w_ds)
for d_id in range(d_is.shape[0]):
    print(d_id)
    p_w_ds[d_id,:] = p_w_ds[d_id,:]/d_is[d_id,0]

def smoothed_kl_dist(p_d1,p_d2,p_ref,mu,d1,d2):
    l1 = mu/(mu+d1)
    l2 = mu/(mu+d2)
    p_d1 = p_d1 * (1-l1) + p_ref * l1
    p_d2 = p_d2 + (1-l2) + p_ref + l2
    dist = np.sum(p_d1 * np.log(p_d1/p_d2))
    return dist

print("Checking distanse")
with open("doc_dist_kl.txt","w") as d_dist:
 for d1_id in range(d_is.shape[0]):
  d_dist.write("%d "%d1_id)
  print("\r%d/%d"%(doc_id,d_is.shape[0]),end = "")
  for d2_id in range(d_is.shape[0]):
      dist = smoothed_kl_dist(p_w_ds[d1_id,:],
                                p_w_ds[d2_id,:],
                                p_w_cs,
                                mu,
                                d_is[d1_id,0],
                                d_is[d2_id,0])
      d_dist.write("%f "%dist)
  d_dist.write("\n")





# def convert_sparse_matrix_to_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return tf.SparseTensor(indices, coo.data, coo.shape)
