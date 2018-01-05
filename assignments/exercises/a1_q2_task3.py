import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import sklearn.model_selection as sk
import pandas as pd
import time
import numpy as np
import csv

# Step 1: Read in data
X, Y = pd.read_csv('heart.csv').iloc[:, 0:9], pd.read_csv('heart.csv').iloc[:, 9:]
X['famhist'] = X['famhist'].map({'Present': 1, 'Absent': 0})

X_train, X_test, Y_train, Y_test = sk.train_test_split(X.values, Y.values, test_size=0.2, random_state=42)
n_sample = X_train.shape[0]
# Step 2: create placeholders for features and labels
X = tf.placeholder(tf.float32, shape=[1, 9])
Y = tf.placeholder(tf.float32)

# Step 3: create weights and bias
weights = tf.Variable(initial_value=tf.random_normal(shape=[9, 1], stddev=0.01), name='weights')
bias = tf.Variable(initial_value=0.0, name='bias')

# Step 4: build model
logits = tf.matmul(X, tf.reshape(weights, (-1, 1))) + bias

# Step 5: define loss function
hyp = tf.sigmoid(logits)
cost0 = Y * tf.log(hyp)
cost1 = (1 - Y) * tf.log(1 - hyp)
cost = (cost0 + cost1) / -n_sample
loss = tf.reduce_sum(cost)

# Step 6: define training op
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        total_loss = 0
        for (x, y) in zip(X_train, Y_train):
            x = np.reshape(x, (1, 9))
            _, loss_poch = sess.run([optimizer, loss], feed_dict={X: x, Y: y[0]})
            total_loss += loss_poch
            #print('Average loss epoch {0}: {1}'.format(i, total_loss / n_sample))
    print("Total time: {0} seconds".format(time.time() - start_time))
    print('Optimization Finished!')
    # test the model
    total_accuracy = 0
    for (x, y) in zip(X_test, Y_test):
        x = np.reshape(x, (1, 9))
        _, lossn, logitsn = sess.run([optimizer, loss, logits], feed_dict={X: x, Y: y[0]})
        pres = tf.sigmoid(logitsn)
        correct_preds = tf.equal(tf.round(pres), y)
        total_accuracy += sess.run(correct_preds)
    print('Accuracy {0}'.format(total_accuracy / Y_test.shape[0]))
