
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from data_manager import DataManager
from data_manager import get_data

SAMPLE_RATE = 16000
SECONDS_OF_AUDIO = 3
N_CLASSES = 50
DROPOUT = 0.75
BATCH_SIZE = 20
LEARNING_RATE = 0.01
PRINT_EVERY = 20
EVAL_EVERY = 200
SAVE_DIR = './checkpoints/'

header, train, val, test, data_dict = get_data(N_CLASSES)

x = tf.placeholder(tf.float32, [None, SAMPLE_RATE * SECONDS_OF_AUDIO])
keep_prob = tf.placeholder(tf.float32) # Dropout

def conv1d(x, W, b, stride=1):
    x = tf.nn.conv1d(x, W, stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool1d(x, k=2):
    x = tf.expand_dims(x, 0)
    x = tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1], padding='SAME') # 1, N, W, C
    x = tf.squeeze(x, squeeze_dims=[0])
    return x

def net(x, weights, biases, dropout):
    x = tf.expand_dims(x, 2) # x is (N, 48000)
    conv0 = conv1d(x, weights['w0'], biases['b0'], stride=200) # (128, 240)

    conv1 = conv1d(conv0, weights['w1'], biases['b1']) # (32, 240)
    conv1 = maxpool1d(conv1, k=4) # (32, 60)
    
    conv2 = conv1d(conv1, weights['w2'], biases['b2']) # (32, 120)
    conv2 = maxpool1d(conv2, k=4) # (32, 15)
    
    fc1 = tf.reshape(conv2, [-1, weights['w3'].get_shape().as_list()[0]]) # (480)
    fc1 = tf.add(tf.matmul(fc1, weights['w3']), biases['b3']) # (100)
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out']) # (50)
    return out

weights = {
    'w0': tf.Variable(tf.random_normal([200, 1, 128])),
    'w1': tf.Variable(tf.random_normal([8, 128, 32])),
    'w2': tf.Variable(tf.random_normal([8, 32, 32])),
    'w3': tf.Variable(tf.random_normal([480, 100])),
    'out': tf.Variable(tf.random_normal([100, N_CLASSES]))
}

biases = {
    'b0': tf.Variable(tf.random_normal([128])),
    'b1': tf.Variable(tf.random_normal([32])),
    'b2': tf.Variable(tf.random_normal([32])),
    'b3': tf.Variable(tf.random_normal([100])),
    'out': tf.Variable(tf.random_normal([N_CLASSES]))
}

coord = tf.train.Coordinator()
data_man_train = DataManager(train, data_dict, coord, SAMPLE_RATE, SECONDS_OF_AUDIO, N_CLASSES)
data_man_val = DataManager(val, data_dict, coord, SAMPLE_RATE, SECONDS_OF_AUDIO, N_CLASSES)

x_batch, y_batch = data_man_train.dequeue(BATCH_SIZE)
x_batch_val, y_batch_val = data_man_val.dequeue(BATCH_SIZE)
pred = net(x_batch, weights, biases, keep_prob)
pred_val = net(x_batch_val, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_batch))
cost_val = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred_val, y_batch_val))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

auc_op = tf.contrib.metrics.streaming_auc(tf.sigmoid(pred), y_batch)
auc_op_val = tf.contrib.metrics.streaming_auc(tf.sigmoid(pred_val), y_batch_val)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.initialize_all_variables())

threads = tf.train.start_queue_runners(sess=sess, coord=coord)
data_man_train.start_threads(sess)
data_man_val.start_threads(sess)

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

try:
    print 'Starting training'
    step = 0
    #start = time.time()
    while True:
        sess.run(optimizer, feed_dict={keep_prob: DROPOUT})
        if step % PRINT_EVERY == 0:
            sess.run(tf.initialize_local_variables())
            loss, auc = sess.run([cost, auc_op], feed_dict={keep_prob: 1})
            print 'Step', step, 'Minibatch loss', loss, 'Minibatch AUC', auc
        if step % EVAL_EVERY == 0:
            sess.run(tf.initialize_local_variables())
            total_loss = 0
            for _ in range(len(val) / BATCH_SIZE):
                loss, auc = sess.run([cost_val, auc_op_val], feed_dict={keep_prob: 1})
                total_loss += loss
            print 'Validation set loss', total_loss / (len(val) / BATCH_SIZE), 'Validation set AUC', auc
            saver.save(sess, SAVE_DIR + 'model', global_step=step)
        step += 1
except:
    pass
finally:
    coord.request_stop()

coord.join(threads)
sess.close()
