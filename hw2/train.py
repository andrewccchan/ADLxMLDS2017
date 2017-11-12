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

def train_network(video_list, graph, model_path, logging_file, params):
    batch_size = params['batch_size']
    num_epoch = params['num_epoch']

    sess = tf.Session()
    # graph['train_op'] = tf.train.AdamOptimizer(params['learning_rate']).minimize(graph['loss'])   

    sess.run(tf.global_variables_initializer())
    model_saver = tf.train.Saver()

    log_file = open(logging_file ,'w')
    valid_interval = 25
    for idx in range(num_epoch):
        print('Epoch: {}'.format(idx))
        batch_gen = BatchGenerator(video_list, batch_size)
        step = 0
        training_loss = 0

        while batch_gen.check_next_batch():
            data = batch_gen.gen_batch()
            ids, x, y, y_mask = list(zip(*data))
            # print(ids)
            # pdb.set_trace()
            # Fill feed_dict
            ###########
            # feed_dict = {graph['x']:x, graph['y']:y, graph['y_mask']:y_mask}
            # loss_, _ = sess.run([graph['loss'], graph['train_op']], feed_dict=feed_dict)
            # training_loss += loss_

            # if step % valid_interval == 0 and step > 0:
            #     msg = 'Step = {}. Training loss = {}.'.format(step, training_loss/valid_interval)
            #     log_file.write(msg + '\n')
            #     print(msg)
            #     training_loss = 0
            ############
            
            step += 1

        for v in video_list:
            v.reset()

        model_saver.save(sess, model_path, global_step=idx)
    log_file.close()

import os
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    fea_path = './data/training_data/feat'
    label_path = './data/training_label.json'
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

    # Save params
    # Read word_idx_map
    with open('word_idx.map', 'rb') as wi:
        word_idx_map = pickle.load(wi)

    # Read labels
    labels = utility.read_labels(label_path)

    # Generate video list
    video_list = utility.gen_video_list(word_idx_map, fea_path, params['sent_len'], labels)
    print(len(video_list))
    # Get serial number
    serial_man = utility.SerialNumberManager('./model', './log', './params', './out')
    model_path, log_path, params_path = serial_man.gen_paths()
    print('[INFO]:\n model_path={}\n log_path={}\n params_path={}\n'.format(model_path, log_path, params_path))

    # Save params
    with open(params_path, 'wb') as pm:
        pickle.dump(params, pm)
    
    # Build training network
    caption_gen = S2VT(params)
    graph = caption_gen.build_train_network()

    # Train network
    train_network(video_list, graph, model_path, log_path, params)
