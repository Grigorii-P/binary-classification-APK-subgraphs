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

a = np.array([[1,2,3],[4,5,2]])

print(np.amin(a))