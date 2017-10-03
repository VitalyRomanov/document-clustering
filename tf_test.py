import tensorflow as tf

i0 = tf.constant(0)
m0 = tf.ones([2, 2])
v0 = tf.Variable([[0,0],[0,0]],dtype=tf.float32)
c = lambda i, m, v: i < 10
# b = lambda i, m, v: [i+1, m, tf.scatter_add(v,1,1.)]
b = lambda i, m, v: [i+1, m, v+1]
r = tf.while_loop(
    c, b, loop_vars=[i0, m0, v0[0,0]])

init = tf.global_variables_initializer()

# ass = tf.assign(v0[1,1],5)

with tf.Session() as sess:
    sess.run(init)
    _,res_m,res_v = sess.run(r)
    # res = sess.run(ass)
    print(res_m)
    print(res_v)
    # print(res)
