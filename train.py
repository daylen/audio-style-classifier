import time
import tensorflow as tf
import os
from data_manager import DataManager
from data_manager import get_data
import argparse
import librosa
import numpy as np

SAMPLE_RATE = 16000
SECONDS_OF_AUDIO = 3
N_CLASSES = 50
DROPOUT = 0.75
BATCH_SIZE = 20
LEARNING_RATE = 0.001
PRINT_EVERY = 20
EVAL_EVERY = 200
STD_DEV = 0.01
SAVE_DIR = './checkpoints3/'

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

def get_end_ops(fname_list, data_dict, coord, weights, biases):
    data_man = DataManager(fname_list, data_dict, coord, SAMPLE_RATE, SECONDS_OF_AUDIO, N_CLASSES, 10*BATCH_SIZE)
    x_batch, y_batch = data_man.dequeue(BATCH_SIZE)
    pred = net(x_batch, weights, biases, keep_prob)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_batch))
    auc_op, update_auc_op = tf.contrib.metrics.streaming_auc(tf.sigmoid(pred), y_batch)

    return data_man, pred, cost, auc_op, update_auc_op

def get_vars():
    weights = {
        'w0': tf.Variable(tf.random_normal([200, 1, 128], stddev=STD_DEV)),
        'w1': tf.Variable(tf.random_normal([8, 128, 32], stddev=STD_DEV)),
        'w2': tf.Variable(tf.random_normal([8, 32, 32], stddev=STD_DEV)),
        'w3': tf.Variable(tf.random_normal([480, 100], stddev=STD_DEV)),
        'out': tf.Variable(tf.random_normal([100, N_CLASSES], stddev=STD_DEV))
    }

    biases = {
        'b0': tf.Variable(tf.random_normal([128], stddev=STD_DEV)),
        'b1': tf.Variable(tf.random_normal([32], stddev=STD_DEV)),
        'b2': tf.Variable(tf.random_normal([32], stddev=STD_DEV)),
        'b3': tf.Variable(tf.random_normal([100], stddev=STD_DEV)),
        'out': tf.Variable(tf.random_normal([N_CLASSES], stddev=STD_DEV))
    }
    return weights, biases

def train():
    header, train, val, test, data_dict = get_data(N_CLASSES)
    print header

    weights, biases = get_vars()

    coord = tf.train.Coordinator()

    data_man_train, pred, cost, auc_op, update_auc_op = get_end_ops(train, data_dict, coord, weights, biases)
    data_man_val, pred_val, cost_val, auc_op_val, update_auc_op_val = get_end_ops(val, data_dict, coord, weights, biases)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    data_man_train.start_threads(sess)
    data_man_val.start_threads(sess)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    try:
        print 'Starting training'
        step = 0
        start = time.time()
        while True:
            sess.run(optimizer, feed_dict={keep_prob: DROPOUT})
            if step % PRINT_EVERY == 0:
                loss, _ = sess.run([cost, update_auc_op], feed_dict={keep_prob: 1})
                print 'Step', step, 'Epochs', float(step) * BATCH_SIZE / len(train), \
                    'Minibatch loss', loss, 'Time', time.time() - start
            if step % EVAL_EVERY == 0 and step != 0:
                total_loss = 0
                for _ in range(len(val) / BATCH_SIZE):
                    loss, _ = sess.run([cost_val, update_auc_op_val], feed_dict={keep_prob: 1})
                    total_loss += loss
                auc, auc_val = sess.run([auc_op, auc_op_val])
                print 'Train set AUC', auc
                print 'Validation set loss', total_loss / (len(val) / BATCH_SIZE), 'Validation set AUC', auc_val
                print ''
                saver.save(sess, SAVE_DIR + 'model', global_step=step)
                sess.run(tf.initialize_local_variables())
            step += 1
    except:
        pass
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def evaluate(model, song_fname, start_in_seconds):
    assert model != None and song_fname != None and start_in_seconds != None
    header, _, _, _, _ = get_data(N_CLASSES)

    weights, biases = get_vars()

    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, model)
    pred = tf.sigmoid(net(x, weights, biases, keep_prob))

    audio = librosa.load(song_fname, sr=SAMPLE_RATE)[0]
    assert start_in_seconds * SAMPLE_RATE < len(audio)
    audio = audio[start_in_seconds * SAMPLE_RATE:start_in_seconds * SAMPLE_RATE + SECONDS_OF_AUDIO * SAMPLE_RATE]
    audio = np.array(audio)
    audio = np.expand_dims(audio, 0)

    result = sess.run(pred, feed_dict={x: audio, keep_prob: 1}).flatten()
    assert len(header) == len(result)

    tag_prob_list = []

    for i in range(N_CLASSES):
        tag_prob_list.append((header[i], result[i]))

    top_tags = sorted(tag_prob_list, key=lambda x: x[1])[::-1][:10]
    for a, b in top_tags:
        print a, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval'])
    parser.add_argument('--model')
    parser.add_argument('--song')
    parser.add_argument('--start_pos', type=int)
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate(args.model, args.song, args.start_pos)
    else:
        raise Exception('invalid mode')

if __name__ == '__main__':
    main()
