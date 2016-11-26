import tensorflow as tf
import numpy as np
import librosa
import threading

class DataManager:
	def __init__(self, fnames, data_dict, coord, sample_rate, seconds_of_audio, n_classes):
		self.fnames = fnames
		self.data_dict = data_dict
		self.coord = coord
		self.sample_rate = sample_rate
		self.seconds_of_audio = seconds_of_audio
		self.x = tf.placeholder(tf.float32, [sample_rate * seconds_of_audio])
		self.y = tf.placeholder(tf.float32, [n_classes])
		self.queue = tf.RandomShuffleQueue(500, 50, ['float32', 'float32'], shapes=[self.x.get_shape(), self.y.get_shape()])
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
				audio = librosa.load(fname, sr=self.sample_rate)[0]
				rand_start_idx = np.random.randint(0, len(audio) - self.sample_rate * self.seconds_of_audio)
				audio = audio[rand_start_idx:rand_start_idx + self.sample_rate * self.seconds_of_audio]
				label = self.data_dict[fname]
				sess.run(self.enqueue_op, feed_dict={self.x: audio, self.y: label})

	def start_threads(self, sess):
		thread = threading.Thread(target=self.thread_main, args=(sess,))
		thread.daemon = True
		thread.start()
		return thread
