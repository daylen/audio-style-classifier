import time
import tensorflow as tf
import os
from data_manager import DataManager
from data_manager import get_data
import argparse
import librosa
import numpy as np
import train

def print_tags(header, result):
    tag_prob_list = []
    for i in range(train.N_CLASSES):
        tag_prob_list.append((header[i], result[i]))
    
    top_tags = sorted(tag_prob_list, key=lambda x: x[1])[::-1]
    for a, b in top_tags[:10]:
        print a, b
    print '...'
    for a, b in top_tags[-11:-1]:
        print a, b

def style_transfer(model, song_fname, start_in_seconds):
    # First we restore the weights from the checkpoint
    header, _, _, _, _ = get_data(train.N_CLASSES, train.MERGE_TAGS, train.SPLIT_RANDOMLY)
    weights, biases = train.get_vars()
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, model)

    print 'loading song'
    audio = librosa.load(song_fname, sr=train.SAMPLE_RATE)[0]
    start_idx = start_in_seconds * train.SAMPLE_RATE
    # Clip to 3 seconds
    clip = audio[start_idx : start_idx + train.SECONDS_OF_AUDIO * train.SAMPLE_RATE]
    clip = np.expand_dims(np.array(clip), 0)
    print clip.shape

    x_var = tf.Variable(clip, name='x')
    pred = train.net(x_var, weights, biases, train.keep_prob) # no sigmoid applied

    # Print the predictions for the clip
    sess.run(x_var.initializer)
    librosa.output.write_wav('./cropped.wav', x_var.eval(session=sess).flatten(), train.SAMPLE_RATE)

    result = sess.run(pred, feed_dict={train.keep_prob: 1}).flatten()
    print_tags(header, result)

    print ''
    target = raw_input('Which tag to boost? ')
    assert target in header
    target_idx = header.index(target)
    target_vec = result.copy()
    # Make the target the largest logit
    target_vec[target_idx] = np.amax(target_vec) + 1
    print 'OK, will try to get logit equal to', target_vec[target_idx]
    target_vec = np.expand_dims(target_vec, 0)
    print target_vec.shape

    # Optimize with respect to input
    cost = tf.reduce_mean(tf.nn.l2_loss(pred - target_vec))
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    opt_op = opt.minimize(cost, var_list=[x_var])

    print 'starting optimization'
    step = 0
    start = time.time()
    while True:
        sess.run(opt_op, feed_dict={train.keep_prob: 1})
        if step % 500 == 0:
            loss = sess.run(cost, feed_dict={train.keep_prob: 1})
            result = sess.run(pred, feed_dict={train.keep_prob: 1}).flatten()
            # print_tags(header, result)
            print 'Logit is currently', result[target_idx]
            print 'Step', step, 'Loss', loss, 'Time', time.time() - start
        if step % 5000 == 0:
            librosa.output.write_wav('./transfer' + str(step) + '.wav', x_var.eval(session=sess).flatten(), train.SAMPLE_RATE)
            print 'saved'
        step += 1
    sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--song', required=True)
    parser.add_argument('--start', required=True)
    args = parser.parse_args()
    style_transfer(args.model, args.song, int(args.start))

if __name__ == '__main__':
    main()