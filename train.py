
# coding: utf-8

# In[1]:

import librosa
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import os


# In[2]:

root_path = "/Users/daylenyang/Downloads/magnatagatune/mp3/"
csv_file_path = root_path + "annotations_final.csv"


# In[3]:

synonyms = [['beat', 'beats'],
            ['chant', 'chanting'],
            ['choir', 'choral'],
            ['classical', 'clasical', 'classic'],
            ['drum', 'drums'],
            ['electro', 'electronic', 'electronica', 'electric'],
            ['fast', 'fast beat', 'quick'],
            ['female', 'female singer', 'female singing', 'female vocals', 'female voice', 'woman', 'woman singing', 'women'],
            ['flute', 'flutes'],
            ['guitar', 'guitars'],
            ['hard', 'hard rock'],
            ['harpsichord', 'harpsicord'],
            ['heavy', 'heavy metal', 'metal'],
            ['horn', 'horns'],
            ['india', 'indian'],
            ['jazz', 'jazzy'],
            ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
            ['no beat', 'no drums'],
            ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
            ['opera', 'operatic'],
            ['orchestra', 'orchestral'],
            ['quiet', 'silence'],
            ['singer', 'singing'],
            ['space', 'spacey'],
            ['string', 'strings'],
            ['synth', 'synthesizer'],
            ['violin', 'violins'],
            ['vocal', 'vocals', 'voice', 'voices'],
            ['strange', 'weird']]


# In[4]:

def get_top_tags(k):
    df = pd.read_csv(csv_file_path, sep='\t')
    del df['mp3_path']
    del df['clip_id']
    sums = np.sum(df, axis=0)
    return map(lambda x: x[0], sorted(sums.iteritems(), key=lambda x: x[1])[::-1][:50])


# In[5]:

get_top_tags(50)


# In[6]:

def get_data_dict(N):
    """
    header: an array of strings
    data_dict: a dictionary with filenames as keys and arrays of integers
    as values. The array contains the values corresponding to the header.
    """
    df = pd.read_csv(csv_file_path, sep='\t')
    df_top_50 = df[get_top_tags(50) + ['mp3_path']]
    df_dict = df_top_50.to_dict('split')
    header = df_dict['columns'][:-1]
    rows = df_dict['data']
    ret_val = {}
    for row in rows:
        fname = row[-1]
        ret_val[os.path.join(root_path, fname)] = np.array(row[:-1], dtype=np.int)
    return header, ret_val
header, data_dict = get_data_dict(50)

baby_set = data_dict.keys()[:50]


# In[7]:

baby_set


# # network definition

# In[8]:

SAMPLE_RATE = 16000
SECONDS_OF_AUDIO = 3
N_CLASSES = 50
DROPOUT = 0.75


# In[9]:

def next_batch(batch_size):
    filenames = baby_set
    print len(filenames)
    filename_queue = tf.train.string_input_producer(filenames)
    
    reader = tf.IdentityReader()
    fname, value = reader.read(filename_queue)
    print fname, value
    
    audio = np.flatten(librosa.load(fname, sr=SAMPLE_RATE)[0])
    print audio.shape
    audio = tf.random_crop(audio, SAMPLE_RATE * SECONDS_OF_AUDIO)
    label = data_dict[fname]
    
    audios, labels = tf.train.shuffle_batch(
        [audio, label],
        batch_size=batch_size,
        num_threads=16,
        capacity=5*batch_size,
        min_after_dequeue=2*batch_size
    )
    
    return audios, labels


# In[13]:

curr_idx = 0
def basic_next_batch(batch_size):
    global curr_idx
    fnames = baby_set[curr_idx:curr_idx + batch_size]
    curr_idx += batch_size
    if curr_idx >= len(baby_set):
        curr_idx = 0
    audios, labels = [], []
    for fname in fnames:
        audio = librosa.load(fname, sr=SAMPLE_RATE)[0]
#         print audio.shape
#        audio = tf.constant(audio)
        #print audio.get_shape()
        #audio = tf.random_crop(audio, [SAMPLE_RATE * SECONDS_OF_AUDIO])
        rand_start_idx = np.random.randint(0, len(audio) - SAMPLE_RATE * SECONDS_OF_AUDIO)
        audio = audio[rand_start_idx:rand_start_idx + SAMPLE_RATE * SECONDS_OF_AUDIO]
        label = data_dict[fname]
        audios.append(audio)
        labels.append(label)
        
    
    audios, labels = np.array(audios), np.array(labels)
    #print audios.shape
    #print labels.shape
    return audios, labels


# In[17]:

BATCH_SIZE = 10
LEARNING_RATE = 0.01
x = tf.placeholder(tf.float32, [None, SAMPLE_RATE * SECONDS_OF_AUDIO])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
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
    x = tf.expand_dims(x, 2)
    conv1 = conv1d(x, weights['w1'], biases['b1'])
    conv1 = maxpool1d(conv1, k=8)
    
    conv2 = conv1d(conv1, weights['w2'], biases['b2'])
    conv2 = maxpool1d(conv2, k=8)
    
    fc1 = tf.reshape(conv2, [-1, weights['w3'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['w3']), biases['b3'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    'w1': tf.Variable(tf.random_normal([8, 1, 32])),
    'w2': tf.Variable(tf.random_normal([8, 32, 32])),
    'w3': tf.Variable(tf.random_normal([SAMPLE_RATE * SECONDS_OF_AUDIO * 32 / 8 / 8, 500])),
    'out': tf.Variable(tf.random_normal([500, N_CLASSES]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([32])),
    'b2': tf.Variable(tf.random_normal([32])),
    'b3': tf.Variable(tf.random_normal([500])),
    'out': tf.Variable(tf.random_normal([N_CLASSES]))
}

pred = net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# todo evaluation

init = tf.initialize_all_variables()

print 'Starting training'
sess = tf.Session()
print 'init variables...'
sess.run(init)
print 'done with init'
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    step = 1
    while step < 100:
        #print step
        batch_x, batch_y = basic_next_batch(BATCH_SIZE)
#         print tf.constant(batch_x).get_shape()
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: DROPOUT})
        if step % 10 == 0:
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
            print 'Step', step, 'Minibatch Loss', loss
        step += 1
    print 'Optimization done'
    coord.request_stop()
except tf.errors.OutOfRangeError:
    print 'done training'
finally:
    coord.request_stop()

coord.join(threads)
sess.close()


# In[ ]:



