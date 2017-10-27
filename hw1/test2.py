#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import utility
from utility import BatchGenerator
import math

# Global variables
# Mapping files
(phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map) = utility.read_map('./data')

def process_phone_buffer(phone_buffer):
    processed = []
    last_char = ''
    # Remove repetitive chars
    for phone in phone_buffer:
        if phone != last_char:
            processed.append(phone)
        last_char = phone
    # Remove leading and tailing 'sil'
    if processed[0] == 'L':
        del processed[0]
    if processed[-1] == 'L':
        del processed[-1]

    return ''.join(processed)

# level 0: output id
# level 1: output phone
# level 2: output reduced
# level 3: output char
def phone_transform(idx, level=0):
    phone = idx_phone_map[idx]
    phone_reduce = phone_reduce_map[phone]
    char = reduce_char_map[phone_reduce]

    if level == 0:
        return idx
    elif level == 1:
        return phone
    elif level == 2:
        return phone_reduce

    return char

def get_speaker_id(phone_id):
    return '_'.join(phone_id.split('_')[0:2])

def output_smoothing(phone_wise, length):
    edited_phone = []
    tmp = []
    for _, p in enumerate(phone_wise):
        if len(tmp) > 0 and p[1] != tmp[-1][1]:
            if len(tmp) > length:
                edited_phone.extend(tmp)
            tmp = []
        tmp.append(p)
    if len(tmp) > length:
        edited_phone.extend(tmp)
    return edited_phone

def output_phone_sequence(phone_wise, path, smooth_len=0):
    if smooth_len > 0:
        phone_wise = output_smoothing(phone_wise, smooth_len)
    phone_buffer = []
    out_file = open(path, 'w')
    out_file.write('id,phone_sequence\n')
    for idx in range(len(phone_wise)):
        phone_buffer.append(phone_transform(phone_wise[idx][1], level=3))

        # check whether write phone buffer to file
        cur_id = get_speaker_id(phone_wise[idx][0])
        nxt_id = None if (idx == len(phone_wise) - 1) else get_speaker_id(phone_wise[idx+1][0])

        if nxt_id is None or cur_id != nxt_id:
            speaker_id = get_speaker_id(phone_wise[idx][0])
            pheon_seq = process_phone_buffer(phone_buffer)
            out_file.write(speaker_id + ',' + pheon_seq + '\n')
            del phone_buffer
            phone_buffer = []

    out_file.close()



def output_phone_wise(phone_wise, file_path, level=0):
    with open(file_path, 'w') as f:
        for idx in range(len(phone_wise)):
            phone_id = phone_wise[idx][0]
            output = phone_transform(phone_wise[idx][1], level)
            f.write(phone_id + ',' + str(output) + '\n')

def predict(speaker_list, model_path, model_name, out_path):
    # Model file path
    meta_file = os.path.join(model_path, model_name+'.meta')
    # checkpt_file = os.path.join(model_path, 'checkpoint')

    #TODO: read hyper-parameters from model file
    # hyper-parameters
    batch_size = 128
    num_steps = 31
    # out_file = open(out_path, 'w')
    # out_file.write('id,phone_sequence\n')

    # output container
    phone_wise = []

    print('Testing on {} speakers'.format(len(speaker_list)))
    with tf.Session() as sess:
        # Load model
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        #sess.run(tf.global_variables_initializer())
        # all_vars = tf.trainable_variables()
        X = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        #keep_prob = tf.get_collection('keep_prob')[0]
        loss = tf.get_collection('total_loss')[0]
        predictions = tf.get_collection('predictions')[0]
        graph = tf.get_default_graph()
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
        print(keep_prob)

        # Init. batch generator
        batch_gen = BatchGenerator(speaker_list, batch_size)
        total_loss = 0

        # Predict phones batch by batch
        while batch_gen.check_next_batch():
            data = batch_gen.gen_batch()
            x = []
            ids = []
            for d in data:
                ids.append(d[0])
                x.append(d[1])
            actual_batch_size = len(x)
            # print('actual_batch_size: ', actual_batch_size)
            dummy_y = np.zeros((actual_batch_size, num_steps))
            feed_dict = {X: x, y: dummy_y, keep_prob: 1.0}
            loss_, predict_ = sess.run([loss, predictions], feed_dict=feed_dict)
            total_loss += loss_

            # print(predict_)
            # Store phone-wise predictions
            prediction_idx = int(math.floor(num_steps / 2))

            for idx in range(actual_batch_size):
                phone_wise.append([ids[idx][prediction_idx], predict_[idx]])

    # Generate phone sequences
    output_phone_wise(phone_wise, os.path.join(out_path, '15_phone_wise_train.out'), level=1)
    output_phone_sequence(phone_wise, os.path.join(out_path, '15_phone_sequence_train.out'), 2)

import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    model_path = './model/cnn_rnn_lstm/'
    model_name = '12-26'
    test_data_mfcc = utility.read_data('./data', 'mfcc', 'test')
    test_data_fbank = utility.read_data('./data', 'fbank', 'test')
    test_data = utility.merge_features(test_data_mfcc, test_data_fbank)
    speaker_list = utility.gen_speaker_list(phone_reduce_map, phone_idx_map, 31, test_data)
    predict(speaker_list, model_path, model_name, './out')
