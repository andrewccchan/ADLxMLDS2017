#!/usr/bin/env python3
import pickle
import numpy as np
import utility_new
import network_new
import tensorflow as tf 
from utility_new import BatchGenerator
from network_new import S2VT
import pdb
import time
import sys

with open('idx_word_new.map', 'rb') as iw:
    idx_word_map = pickle.load(iw)

def isascii(s):
    return len(s) == len(s.encode())

def proc_words(sent):
    buf = []
    for i in range(0, len(sent)):
        word = idx_word_map[sent[i]]
        if isascii(word) and sent[i] != 0 and sent[i] != 1 and sent[i] != 2:
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
            #pdb.set_trace()
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
    #os.environ['CUDA_VISIBLE_DEVICES']='1'
    data_path = sys.argv[1]
    # fea_path = './data/testing_data/feat'
    fea_path = os.path.join(data_path, 'testing_data/feat')
    # label_path = './data/testing_label.json'
    label_path = os.path.join(data_path, 'testing_label.json')
    # Parameters
    params = dict (
        state_size = 700,
        dict_size = 4000,
        batch_size = 128,
        num_frame = 80,
        feat_size = 4096,
        num_epoch = 25,
        learning_rate = 0.001,
        sent_len = 20
    )


    # Read word_idx_map
    with open('word_idx_new.map', 'rb') as wi:
        word_idx_map = pickle.load(wi)
    with open('bias_init', 'rb') as bi:
        bias_init = pickle.load(bi)
    params['bias_init_vector'] = bias_init

    # Read labels
    labels = utility_new.read_labels(label_path)

    # Generate video list
    video_list = utility_new.gen_video_list(word_idx_map, fea_path, params['sent_len'])
    
    # Get serial number
    # serial_man = utility_new.SerialNumberManager('./model', './log', './params', './out')
    # model_path, params_path, out_path = serial_man.get_paths('96-209')
    model_path = './best_model'
    out_path = sys.argv[2]
    # print('[INFO]:\n model_path={}\n params_path={}\n'.format(model_path, params_path))
    
    # Build training network
    caption_gen = S2VT(params)
    graph = caption_gen.build_test_network()

    # Train network
    predict(video_list, graph, model_path, out_path, params)

    review_feat_path = os.path.join(data_path, 'peer_review/feat')
    review_list = os.path.join(data_path, 'peer_review_id.txt')
    with open(review_list) as rl:
        review_files = [l.rstrip()+'.npy' for l in rl]

    review_video_list = utility_new.gen_video_list_review(word_idx_map, review_feat_path, review_files, params['sent_len'])
    
    review_out_path = sys.argv[3]
    predict(review_video_list, graph, model_path, review_out_path, params)
    