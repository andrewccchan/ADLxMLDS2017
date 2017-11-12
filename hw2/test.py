#!/usr/bin/env python3
import pickle
import numpy as np
import utility
import network
import tensorflow as tf
from utility import BatchGenerator
from network import S2VT
import pdb
import time
import sys

with open('idx_word.map', 'rb') as iw:
    idx_word_map = pickle.load(iw)

def isascii(s):
    return len(s) == len(s.encode())

def proc_words(sent):
    buf = []
    for i in range(1, len(sent)):
        word = idx_word_map[sent[i]]
        if isascii(word) and sent[i] != 0 and sent[i] != 1:
            buf.append(word)
    return ' '.join(buf)

def predict(video_list, graph, model_path, output_path, params):
    batch_size = params['batch_size']
    num_epoch = params['num_epoch']

    sess = tf.Session()
    # graph['train_op'] = tf.train.AdamOptimizer(params['learning_rate']).minimize(graph['loss'])   

    # sess.run(tf.global_variables_initializer())
    model_saver = tf.train.Saver()
    model_saver.restore(sess, model_path)
    
    out_file = open(output_path, 'w')
    for idx in range(1):
        print('Epoch: {}'.format(idx))
        batch_gen = BatchGenerator(video_list, batch_size)
        step = 0
        training_loss = 0

        while batch_gen.check_next_batch():
            data = batch_gen.gen_batch()
            ids, x = list(zip(*data))
            # Fill feed_dict
            feed_dict = {graph['x']:x}
            outputs = sess.run(graph['outputs'], feed_dict=feed_dict)

            outputs = np.transpose(outputs).tolist()
            for i in range(len(ids)):
                # pdb.set_trace()
                out_file.write(ids[i] + ',' + proc_words(outputs[i]) + '\n')

    out_file.close()

import os
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    #fea_path = './data/testing_data/feat'
    fea_path = os.path.join(sys.argv[1], 'testing_data/feat')
    print(fea_path)
    #label_path = './data/testing_label.json'
    # Parameters
    params = dict (
        state_size = 256,
        dict_size = 6349,
        batch_size = 128,
        num_frame = 80,
        feat_size = 4096,
        num_epoch = 25,
        learning_rate = 0.0001,
        sent_len = 20
    )


    # Read word_idx_map
    with open('word_idx.map', 'rb') as wi:
        word_idx_map = pickle.load(wi)

    # Read labels
    #labels = utility.read_labels(label_path)

    # Generate video list
    video_list = utility.gen_video_list(word_idx_map, fea_path, params['sent_len'])
    print(len(video_list)) 
    # Get serial number
    #serial_man = utility.SerialNumberManager('./model', './log', './params', './out')
    #model_path, params_path, out_path = serial_man.get_paths()
    #print('[INFO]:\n model_path={}\n params_path={}\n'.format(model_path, params_path))

    # Build training network
    caption_gen = S2VT(params)
    graph = caption_gen.build_test_network()
    
    model_path = './1-1'
    out_path = sys.argv[2]
    # Train network
    predict(video_list, graph, model_path, out_path, params)
