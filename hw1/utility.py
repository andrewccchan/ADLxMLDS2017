#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import math

class Speaker:
    X = None
    y = None
    train_mode = False
    ids = None
    num_steps = -1
    begin = 0

    def __init__(self, num_steps, X, y=None):
        half_step = int(math.floor(num_steps / 2))
        self.X = X if type(X).__module__ == np.__name__ else np.asarray(X)
        self.train_mode = True if type(y).__module__ == np.__name__ else False
        (m, n) = self.X.shape
        # pad X with zeros
        padding_x = np.zeros((half_step - 1, n))
        X_pad = np.vstack((padding_x, X, padding_x))
        self.ids = X_pad[:, 0]
        self.X = X_pad[:, 1:]
        assert(self.X.shape[0] == m + 2*(half_step - 1))
        # pad y if y is given
        if self.train_mode:
            padding_y = np.ones((half_step - 1)) * 37
            self.y = np.concatenate([padding_y, y, padding_y])
        self.num_steps = num_steps
        self.begin = 0

    def has_next(self):
        return True if self.begin < self.X.shape[0] - self.num_steps + 1 else False

    # Return one piece of data at a time
    def gen_data(self):
        start = self.begin
        end = start + self.num_steps
        self.begin += 1
        if (end > self.X.shape[0]):
            print('{}, {}'.format(end, self.X.shape[0]))
        assert(end <= self.X.shape[0])
        if self.train_mode:
            return (self.ids[start:end], self.X[start:end, :], self.y[start:end])
        else:
            return (self.ids[start:end], self.X[start:end, :])

    # Reset to default values
    def reset(self):
        self.begin = 0

class BatchGenerator:
    speaker_list = []
    batch_size = 32
    speaker_cnt = 0
    cur_spk = None
    has_next_batch = True

    def __init__(self, speaker_list, batch_size=32):
        self.speaker_list = speaker_list
        self.speaker_cnt = 0
        self.batch_size = batch_size
        self.cur_spk = self.speaker_list[self.speaker_cnt]

    def has_next_speaker(self):
        return True if self.speaker_cnt < len(self.speaker_list) - 1 else False

    def check_next_batch(self):
        return self.has_next_batch

    def gen_batch(self):
        data_cnt = 0
        batch = []
        while data_cnt < self.batch_size:
            if self.cur_spk.has_next():
                # print("case 1")
                batch.append(self.cur_spk.gen_data())
            elif not self.cur_spk.has_next() and self.has_next_speaker():
                # print("case 2")
                self.speaker_cnt += 1
                self.cur_spk = self.speaker_list[self.speaker_cnt]
                batch.append(self.cur_spk.gen_data())
            # Even if cur_spk has next in this iter., has_next_batch still needs to be checked
            if not self.cur_spk.has_next() and not self.has_next_speaker():
                # print('case 3')
                self.has_next_batch = False
                break
            data_cnt += 1
        return batch

# Read faetures
def read_data(data_path, fea_type, mode="train"):
    """Load ark data from `path`"""
    file_path = os.path.join(data_path, fea_type, mode + ".ark")
    print("Reading {} data from {}".format(mode, file_path))
    df = pd.read_csv(file_path, header=None, sep=' ')
    return df.values

# Read labels
def read_train_labels(file_path):
    print("Reading training data labels")
    df = pd.read_csv(file_path, header=None, sep=',')
    return np.asarray(df.values)

# parse input data by speaker and spair data with labels if labels are given
def gen_speaker_list(phone_reduce_map, phone_idx_map, num_steps, data, labels=None):
    print('Generating speaker list')
    speakers = []

    train_mode = type(labels).__module__ == np.__name__

    if train_mode:
        # Lablel dictionary
        label_dict = {labels[i, 0] : labels[i, 1] for i in range(labels.shape[0])}

    tmp_X = []
    tmp_y = []
    last_speaker = ""
    for i in range(data.shape[0]):
        fea_id = data[i, 0]
        cur_speaker = '_'.join(fea_id.split('_')[0:2])

        if cur_speaker != last_speaker and len(tmp_X) > 0:
            if train_mode:
                speakers.append(Speaker(num_steps, np.asarray(tmp_X), np.asarray(tmp_y)))
                del tmp_X
                del tmp_y
                tmp_X = []
                tmp_y = []
            else:
                speakers.append(Speaker(num_steps, np.asarray(tmp_X)))
                del tmp_X
                tmp_X = []
        last_speaker = cur_speaker
        tmp_X.append(data[i, :])
        if train_mode:
            tmp_y.append(phone_idx_map[phone_reduce_map[label_dict[fea_id]]])

    # add the last speaker to the list
    if train_mode:
        speakers.append(Speaker(num_steps, np.asarray(tmp_X), np.asarray(tmp_y)))
    else:
        speakers.append(Speaker(num_steps, np.asarray(tmp_X)))

    return speakers


def pair_data_label(data, labels, phone_idx_map):
    print('Pairing training data and labels')
    X = []
    y = []
    # Create labels dictionary
    label_dict = {labels[i, 0] : labels[i, 1] for i in range(labels.shape[0])}
    for i in range(data.shape[0]):
        instance_id = data[i, 0]
        X.append(data[i, 1:])
        y.append(phone_idx_map[label_dict[instance_id]])
    return (X, y)

def read_map(path):
    print('Reading map file')
    phone_idx_map = dict()
    idx_phone_map = dict()
    idx_char_map = dict()
    phone_reduce_map = dict()
    reduce_char_map = dict()

    # read 48phone_char.map
    with open(os.path.join(path, '48phone_char.map')) as char_map_file:
        for l in char_map_file:
            fields = l.strip('\n').split('\t')
            phone_idx_map[fields[0]] = int(fields[1])
            idx_phone_map[int(fields[1])] = fields[0]
            idx_char_map[int(fields[1])] = fields[2]

    # read 48_39.map
    with open(os.path.join(path, 'phones', '48_39.map')) as reduce_map_file:
        for l in reduce_map_file:
            fields = l.strip('\n').split('\t')
            phone_reduce_map[fields[0]] = fields[1]
            reduce_char_map[fields[1]] = idx_char_map[phone_idx_map[fields[1]]]
    return (phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map)

def merge_features(mfcc, fbank):
    data = []
    for idx, fea in enumerate(mfcc):
        data.append(np.concatenate((fea, fbank[idx][1:])))
    return np.asarray(data)