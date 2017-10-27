#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import utility
import network
from utility import BatchGenerator

def split_train_test(data, test_ratio):
    data_len = len(data)
    test_len = int(data_len * test_ratio)
    np.random.shuffle(data)
    return (data[test_len:], data[:test_len])

def eval_network(sess, test_speaker, graph, params, max_speaker=None):
    # print('Validating network...')
    batch_size = params['batch_size']
    # num_steps = params['num_steps']
    step_cnt = 0
    total_loss = 0

    # If max_speaker is specified, randomly sample `max_speakers` for validation
    if max_speaker != None:
        indices = np.random.choice(range(len(test_speaker)), max_speaker, replace=False)
        sampled_spk = [test_speaker[i] for i in indices]
    else:
        sampled_spk = test_speaker

    # Batch generator
    batch_gen = BatchGenerator(sampled_spk, batch_size)

    while batch_gen.check_next_batch():
        data = batch_gen.gen_batch()
        x = []
        y = []
        ids = []
        for d in data:
            ids.append(d[0])
            x.append(d[1])
            y.append(d[2])
        feed_dict = {graph['x']:x, graph['y']:y, graph['keep_prob']:1.0}
        loss_, _ = sess.run([graph['total_loss'],
                graph['train_step']], feed_dict=feed_dict)
        total_loss += loss_
        step_cnt += 1

    for s in sampled_spk:
        s.reset()

    print('Validation loss = {}'.format(total_loss / step_cnt))
    return total_loss / step_cnt

def train_network(speaker_list, graph, params, model_path, logging_file):
    batch_size = params['batch_size']
    # num_steps = params['num_steps']
    num_epochs = params['num_epochs']

    print('Tatal #sample=', len(speaker_list))
    # Split training and validation set
    (train_speaker, test_speaker) = split_train_test(speaker_list, 0.1)
    print('Training on {} samples, testing on {} samples'.format(len(train_speaker), len(test_speaker)))

    # Begin tensorflow training session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    training_losses = []

    # Open logging file
    log_file = open(logging_file, 'w')

    # Init. model saver
    model_saver = tf.train.Saver()
    
    for idx in range(num_epochs):
        print('Epoch: {}'.format(idx))
        batch_gen = BatchGenerator(train_speaker, batch_size)
        step = 0
        training_loss = 0
        training_state_fw = None
        training_state_bw = None
        
        while batch_gen.check_next_batch():
            data = batch_gen.gen_batch()
            x = []
            y = []
            ids = []
            for d in data:
                ids.append(d[0])
                x.append(d[1])
                y.append(d[2])
            # Fill feed_dict
            feed_dict = {graph['x']:x, graph['y']:y, graph['keep_prob']:0.8}
            if training_state_fw is not None:
                graph['init_state_fw'] = training_state_fw
                graph['init_state_bw'] = training_state_bw
                
            training_loss_, training_state_fw, training_state_bw, _ = \
            sess.run([graph['total_loss'],
                    graph['final_state_fw'],
                    graph['final_state_bw'],
                    graph['train_step']], feed_dict=feed_dict)
            training_loss += training_loss_

            # Validation
            if step % 500 == 0 and step > 0:
                print("Training loss at step", step, "=", training_loss/500)
                training_losses.append(training_loss/500)
                validation_loss = eval_network(sess, test_speaker, graph, params, 10)
                log_file.write(str(training_loss/500) + ',' + str(validation_loss) + '\n')
                training_loss = 0
                
            step += 1

        # save checkpoint
        model_saver.save(sess, model_path, global_step=idx)
        # Reset speakers in list
        for s in speaker_list:
            s.reset()

    # Save trained model
    print('Models saved to ', model_path)
    # print('Saving model to ', model_path)
    # model_saver.save(sess, model_path, )

    return sess, training_losses

import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    params = dict(
        feature_size = 108,
        num_steps = 31, # window size. Can only be a singluar number
        num_classes = 48,
        cnn_filter_size = 3, # 3x3 filter 
        cnn_layer_num = 3, # number of conv. layers
        cnn_filter_num = [32, 32, 32], # 32 filters for each layer
        cnn_pool_size = [2, 1, 1], # only pooling in the first layer
        fc_layer_size = [640, 512, 256], # three layer fully-connected layer
        rnn_state_size = 256, 
        learning_rate = 1e-4,
        batch_size = 128,
        num_epochs = 50, # maximal number of epochs
    )

    (phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map) = utility.read_map('./data')
    mfcc_feature = utility.read_data('./data', 'mfcc', 'train')
    fbank_feature = utility.read_data('./data', 'fbank', 'train')
    data = utility.merge_features(mfcc_feature, fbank_feature)
    # print(data[0])
    labels = utility.read_train_labels('./data/train.lab')
    speaker_list = utility.gen_speaker_list(phone_reduce_map, phone_idx_map, params['num_steps'], data, labels)
    # (X, y) = utility.pair_data_label(raw_data, labels, phone_idx_map)
    model_path = './model/cnn_rnn_lstm/10'
    logging_file = './log/cnn_rnn_lstm/10'



    # Building network
    graph = network.build_cnn_lstm_graph(params)

    # Training
    sess, training_losses = train_network(speaker_list, graph, params, model_path, logging_file)
