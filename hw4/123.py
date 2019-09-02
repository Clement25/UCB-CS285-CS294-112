import numpy as np
import tensorflow as tf

""" a = np.array([[2,0,3,0,0,1]])
b = np.nonzero(a)   # return indices of non-zero elements in matrix a
print(a,b) """

def foo(a, b):
    for i in range(20):
        yield a[i * 5:i * 5 + 5], b[i * 5:i * 5 + 5]

# a = np.arange(100)
# b = np.arange(50, 150)
# gen = foo(a, b)
# for i in range(20):
#     x, y = next(gen)
#     print(x,y)


a = tf.random_uniform(shape=[3,2], minval=1.0, maxval=2.0)
with tf.Session() as sess:
    print(sess.run(a))