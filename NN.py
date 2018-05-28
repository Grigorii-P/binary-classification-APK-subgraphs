import tensorflow as tf
from utils import *
from sklearn.model_selection import train_test_split
import math,logging
from time import time


X, Y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.03, random_state=42)
X_test, y_test = np.array(X_test), np.array(y_test)
x_train_acc, x_test_acc = np.array(X_train), np.array(X_test)
y_train_acc, y_test_acc = np.reshape(np.array(y_train), newshape=(len(y_train),1)), np.reshape(np.array(y_test), newshape=(len(y_test),1))

num_subgraphs = len(X_train[0])
embedding_size = 20
hidden_1 = 50
hidden_2 = 10
bias_value = 0.1
learning_rate = 0.01
epochs = 15
num_batch = 64
beta = 0.01


input = tf.placeholder(tf.float32, shape=([None, num_subgraphs]), name='input_batch')
target = tf.placeholder(tf.float32, shape=([None, 1]), name='label')

subgraph_embeddings = tf.Variable(
        tf.random_normal([num_subgraphs, embedding_size]), name='subgraph_embeddings')

bias = tf.constant(bias_value, shape=[embedding_size], name='bias')

graph_embeddings = tf.add(tf.matmul(input, subgraph_embeddings), bias, name='graph_embeddings')

dense_1 = tf.layers.dense(inputs=graph_embeddings, units=hidden_1, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense_1, rate=0.4)
dense_2 = tf.layers.dense(inputs=dropout, units=hidden_2, activation=tf.nn.relu)

output = tf.layers.dense(inputs=dense_2, units=1, activation=tf.nn.sigmoid, name='output')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=target), name='loss')
# regularizer = tf.nn.l2_loss(subgraph_embeddings)
# loss = tf.reduce_mean(loss + beta * regularizer)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

predict = tf.round(output, name='predict')

saver = tf.train.Saver()

# norm = tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims=True))
# normalized_embeddings = tf.Variable(graph_embeddings / norm, name='normalized_embeddings')

# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#TODO early stopping
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("output", sess.graph)
    writer.close()
    for epoch in range(epochs):
        for x, y in batch(X_train,y_train,num_batch):
            #TODO don't take the whole test dataset when dealing with 800k
            loss_val, _ = sess.run([loss, train_op], feed_dict={input: x, target: y})

            train_accuracy = np.mean(y_train_acc ==
                                     sess.run(predict, feed_dict={input: x_train_acc, target: y_train_acc}))
            test_accuracy = np.mean(y_test_acc ==
                                     sess.run(predict, feed_dict={input: x_test_acc, target: y_test_acc}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%, loss_val = %.2f"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy, loss_val))

    print(np.amax(sess.run(subgraph_embeddings)), np.amin(sess.run(subgraph_embeddings)))
    saver.save(sess, "/home/pogorelov/model_tf/model")

#TODO change paths
