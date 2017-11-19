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
import random

def isascii(s):
    return len(s) == len(s.encode())

def proc_words(sent):
    buf = []
    for i in range(0, len(sent)):
        word = idx_word_map[sent[i]]
        if isascii(word) and sent[i] != 0 and sent[i] != 1:
            buf.append(word)
    return ' '.join(buf)

def split_train_test(data, test_ratio):
    data_len = len(data)
    test_len = int(data_len * test_ratio)
    np.random.shuffle(data)
    return (data[test_len:], data[:test_len])

def eval_network(sess, test_instance, graph, params, max_speaker=None):
    # print('Validating network...')
    batch_size = params['batch_size']
    # num_steps = params['num_steps']
    step_cnt = 0
    total_loss = 0

    # If max_speaker is specified, randomly sample `max_speakers` for validation
    if max_speaker != None:
        indices = np.random.choice(range(len(test_instance)), max_speaker, replace=False)
        sample_inst = [test_instance[i] for i in indices]
    else:
        sample_inst = test_instance

    # Batch generator
    batch_gen = BatchGenerator(sample_inst, batch_size)

    while batch_gen.check_next_batch():
        data = batch_gen.gen_batch()
        ids, x, y, y_mask = list(zip(*data))
        feed_dict = {graph['x']:x, graph['y']:y, graph['y_mask']:y_mask}
        loss_, _ = sess.run([graph['loss'],
                graph['train_op']], feed_dict=feed_dict)
        total_loss += loss_
        step_cnt += 1

    for s in sample_inst:
        s.reset()

    print('Validation loss = {}'.format(total_loss / step_cnt))
    return total_loss / step_cnt

def train_network(video_list, graph, model_path, logging_file, idx_word_map, params):
    batch_size = params['batch_size']
    num_epoch = params['num_epoch']
    (train_video, test_video) = split_train_test(video_list, 0.1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #sess = tf.Session()
    # graph['train_op'] = tf.train.AdamOptimizer(params['learning_rate']).minimize(graph['loss'])   

    sess.run(tf.global_variables_initializer())
    model_saver = tf.train.Saver()

    log_file = open(logging_file ,'w')
    valid_interval = 25
    view_idx = list(range(5))

    for idx in range(num_epoch):
        msg = '\n\nEpoch: {}'.format(idx)
        log_file.write(msg + '\n')
        print(msg)
        random.shuffle(train_video)
        batch_gen = BatchGenerator(train_video, batch_size)
        step = 0
        training_loss = 0
        first = True
        while batch_gen.check_next_batch():
            data = batch_gen.gen_batch()
            ids, x, y, y_mask = list(zip(*data))
            
            # print(ids)
            # Fill feed_dict
            feed_dict = {graph['x']:x, graph['y']:y, graph['y_mask']:y_mask}
            loss_, _, outputs = sess.run([graph['loss'], graph['train_op'], graph['outputs']], feed_dict=feed_dict)
            training_loss += loss_

            if step % valid_interval == 0 and step > 0:
                # Reprot training loss
                msg = '\nStep = {}. Training loss = {}.'.format(step, training_loss/valid_interval)
                log_file.write(msg + '\n')
                # Validation
                eval_loss = eval_network(sess, test_video, graph, params, 10)
                msg += ' Validation loss = {}\n'.format(eval_loss)
                print(msg)

                training_loss = 0
                for vi in view_idx:
                    pre = []
                    msg = proc_words(y[vi])
                    for ii in range(19):
                        pre.append(np.argmax(outputs[ii][vi][:]))
                    msg = msg + '=>' + proc_words(pre)
                    log_file.write(msg + '\n')
                    print(msg)
            step += 1

        for v in train_video:
            v.reset()

        model_saver.save(sess, model_path, global_step=idx)
    log_file.close()

import os
import sys
if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES']='1'
    data_path = sys.argv[1]
    fea_path = os.path.join(data_path, 'testing_data/feat')
    label_path = os.path.join(data_path, 'testing_label.json')

    # fea_path = '/data/MSVD/training_data/feat'
    # label_path = './data/training_label.json'
    # Parameters
    params = dict (
        state_size = 700,
        dict_size = 4000,
        batch_size = 52,
        num_frame = 80,
        feat_size = 4096,
        num_epoch = 200,
        learning_rate = 0.0001,
        sent_len = 20
    )

    # Save params
    # Read word_idx_map
    with open('word_idx_new.map', 'rb') as wi:
        word_idx_map = pickle.load(wi)
    with open('idx_word_new.map', 'rb') as iw:
        idx_word_map = pickle.load(iw)
    with open('bias_init', 'rb') as bi:
        bias_init = pickle.load(bi)
    params['bias_init_vector'] = bias_init

    # Read labels
    labels = utility_new.read_labels(label_path)

    # Generate video list
    video_list = utility_new.gen_video_list(word_idx_map, fea_path, params['sent_len'], labels)

    # Get serial number
    # serial_man = utility_new.SerialNumberManager('./model', './log', './params', './out')
    # model_path, log_path, params_path = serial_man.gen_paths()
    # print('[INFO]:\n model_path={}\n log_path={}\n params_path={}\n'.format(model_path, log_path, params_path))

    # Save params
    # with open(params_path, 'wb') as pm:
    #     pickle.dump(params, pm)
    model_path = './best_model'
    # Build training network
    caption_gen = S2VT(params)
    graph = caption_gen.build_train_network()

    log_path = './best_model_log'
    # Train network
    train_network(video_list, graph, model_path, log_path, idx_word_map, params)
