# import tensorflow as tf
# biases = tf.constant(1, shape=[5])
# init_op = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op) #execute init_op
#     #print the random values that we sample
#     print (sess.run(biases))
from utils import batch, get_data
import numpy as np
import pandas as pd
import tensorflow as tf

x = tf.constant([0.3,0.6,0.49,0.51])
r = tf.round(x)

with tf.Session() as sess:
    print(sess.run(r))

print()