import tensorflow as tf
import numpy as np
import librosa
import threading
import random
import pandas as pd
import os

root_path = "/Users/daylenyang/Downloads/magnatagatune/mp3/"
csv_file_path = root_path + "annotations_final.csv"

train_prefixes = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b'])
val_prefixes = set(['c'])
test_prefixes = set(['d', 'e', 'f'])

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

def _get_top_tags(df, N):
    sums = np.sum(df[df.columns.difference(['mp3_path', 'clip_id'])], axis=0)
    return map(lambda x: x[0], sorted(sums.iteritems(), key=lambda x: x[1])[::-1][:N])

def _merge_tags(df):
	print df.shape, 'pre-merge'
	for arr in synonyms:
		canonical = arr[0]
		for syn in arr[1:]:
			df[canonical] += df[syn]
			df.drop(syn, axis=1, inplace=True)
	print df.shape, 'post-merge'

def _get_data_dict(N, merge=False):
    """
    header: an array of strings
    data_dict: a dictionary with filenames as keys and arrays of integers
    as values. The array contains the values corresponding to the header.
    """
    df = pd.read_csv(csv_file_path, sep='\t')
    if merge:
    	_merge_tags(df)
    top_tags = _get_top_tags(df, N)
    df_top_50 = df[top_tags + ['mp3_path']]
    df_dict = df_top_50.to_dict('split')
    header = df_dict['columns'][:-1]
    rows = df_dict['data']
    ret_val = {}
    for row in rows:
        fname = row[-1]
        ret_val[os.path.join(root_path, fname)] = np.array(row[:-1], dtype=np.int)
    return header, ret_val


def _get_train_val_test_fname_arrs(all_fnames):
	train = []
	val = []
	test = []
	for fname in all_fnames:
		char = fname[len(root_path):len(root_path)+1]
		if char in train_prefixes:
			train.append(fname)
		elif char in val_prefixes:
			val.append(fname)
		elif char in test_prefixes:
			test.append(fname)
		else:
			raise Exception('Could not parse filename')
	return train, val, test

def get_data(N, merge=False):
	header, data_dict = _get_data_dict(N, merge=merge)
	train, val, test = _get_train_val_test_fname_arrs(data_dict.keys())
	random.shuffle(train)
	random.shuffle(val)
	random.shuffle(test)
	print len(train), 'train'
	print len(val), 'val'
	print len(test), 'test'
	return header, train, val, test, data_dict

class DataManager:
	def __init__(self, fnames, data_dict, coord, sample_rate, seconds_of_audio, n_classes, queue_capacity):
		self.fnames = fnames
		self.data_dict = data_dict
		self.coord = coord
		self.sample_rate = sample_rate
		self.seconds_of_audio = seconds_of_audio
		self.x = tf.placeholder(tf.float32, [sample_rate * seconds_of_audio])
		self.y = tf.placeholder(tf.float32, [n_classes])
		self.queue = tf.FIFOQueue(queue_capacity, ['float32', 'float32'], shapes=[self.x.get_shape(), self.y.get_shape()])
		self.enqueue_op = self.queue.enqueue([self.x, self.y])

	def dequeue(self, N):
		return self.queue.dequeue_many(N)

	def thread_main(self, sess):
		stop = False
		while not stop:
			for fname in self.fnames:
				if self.coord.should_stop():
					stop = True
					break
				try:
					audio = librosa.load(fname, sr=self.sample_rate)[0]
				except:
					print 'could not read file', fname, 'skipping...'
					continue
				rand_start_idx = np.random.randint(0, len(audio) - self.sample_rate * self.seconds_of_audio)
				audio = audio[rand_start_idx:rand_start_idx + self.sample_rate * self.seconds_of_audio]
				label = self.data_dict[fname]
				sess.run(self.enqueue_op, feed_dict={self.x: audio, self.y: label})

	def start_threads(self, sess):
		thread = threading.Thread(target=self.thread_main, args=(sess,))
		thread.daemon = True
		thread.start()
		return thread
