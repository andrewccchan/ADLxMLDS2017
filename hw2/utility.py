"""
Utilities for the program
"""
import numpy as np
import os
import pandas as pd
import pdb

class Video:
    def __init__(self, fea, vid_id, *captions_arg):
        self.fea = fea
        self.vid_id = vid_id
        self.captions = captions_arg[0] if len(captions_arg) == 2 else None
        self.mask = captions_arg[1] if len(captions_arg) == 2 else None
        self.caption_idx = 0 # pointer of current caption
        self.train_mode = len(captions_arg) == 2

    def gen_data(self):
        # if self.caption_idx >= len(self.captions):
        #     raise ValueError('Caption list out of range')
        if self.train_mode:
            x = self.fea
            y = self.captions[self.caption_idx]
            mask = self.mask[self.caption_idx]
            self.caption_idx += 1
            return (self.vid_id, x, y, mask)
        else:
            self.caption_idx += 1
            return (self.vid_id, self.fea)
    
    def has_next(self):
        if self.train_mode:
            return self.caption_idx < len(self.captions)
        else:
            return self.caption_idx == 0

    def reset(self):
        self.caption_idx = 0

class BatchGenerator:
    def __init__(self, video_list, batch_size=32):
        self.video_list = video_list
        self.video_cnt = 0
        self.batch_size = batch_size
        self.cur_video = self.video_list[self.video_cnt]
        self.has_next_batch = True

    def has_next_video(self):
        return self.video_cnt < len(self.video_list) - 1

    def check_next_batch(self):
        return self.has_next_batch

    def gen_batch(self):
        data_cnt = 0
        batch = []
        while data_cnt < self.batch_size:
            if self.cur_video.has_next():
                batch.append(self.cur_video.gen_data())
            elif not self.cur_video.has_next() and self.has_next_video():
                self.video_cnt += 1
                self.cur_video = self.video_list[self.video_cnt]
                batch.append(self.cur_video.gen_data())
            else:
                self.has_next_batch = False
            data_cnt += 1
        return batch

def read_labels(label_file):
    """
    Read the label in json format
    """
    return pd.read_json(label_file)

def prcoess_caption(cap, sent_len, word_idx_map):
    """
    Accept a string of caption, and return the pre-processed list of words
    """
    cap = '<bos> ' + cap
    cap = cap + ' <eos>' 
    cap = cap.replace('.', '')
    cap = cap.replace(',', '')
    cap = cap.replace('"', '')
    cap = cap.replace('\n', '')
    cap = cap.replace('?', '')
    cap = cap.replace('!', '')
    cap = cap.replace('\\', '')
    cap = cap.replace('/', '')
    
    word_list = cap.lower().split(' ')
    ret = []
    mask = []
    for idx in range(sent_len):
        if idx >= len(word_list):
            ret.append(word_idx_map['<eos>'])
            mask.append(0)
        else:
            ret.append(word_idx_map[word_list[idx]])
            mask.append(1)
    return ret, mask


def gen_video_list(word_idx_map, fea_path, sent_len, *labels):
    """
    Generate a list of videos
    """

    video_list = []
    fea_list = [i for i in os.listdir(fea_path) if i[-3:]=='npy']
    id_list = [f[:-4] for f in fea_list]
    n_video = len(id_list)
    

    test_vid = ['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi', 'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi', 'tJHUH9tpqPg_113_118.avi']
    if len(labels) == 1:
        for idx in range(n_video):
            v = np.load(os.path.join(fea_path, fea_list[idx]))
            id_idx_dict = {val:idx for idx, val in enumerate(labels[0]['id'])}
            tmp = labels[0]['caption'][id_idx_dict[id_list[idx]]]
            caption_cvt, mask = list(zip(*[prcoess_caption(cap, sent_len, word_idx_map) for cap in tmp]))
            # pdb.set_trace()            
            video_list.append(Video(v, id_list[idx], caption_cvt, mask))
    else:
        for idx in range(n_video):        
            if id_list[idx] not in test_vid:
                continue
            v = np.load(os.path.join(fea_path, fea_list[idx]))        
            video_list.append(Video(v, id_list[idx]))
    
    return video_list

class SerialNumberManager:
    """
    Manager for serial number generation
    """
    def __init__(self, model_base_path, log_base_path, params_base_path, out_base_path):
        self.model_base_path = model_base_path
        self.log_base_path = log_base_path
        self.params_base_path = params_base_path
        self.out_base_path = out_base_path
    
    def list_model_files(self):
        major_num = []
        minor_num = []
        for f in os.listdir(self.model_base_path):
            if f[0] != '.' and f != 'checkpoint':
                major_num.append(int(f.split('-')[0]))
                minor_num.append(int(f.split('-')[1].split('.')[0]))
        return major_num, minor_num

    def get_max_serial(self):
        model_major, model_minor = self.list_model_files()
        log_files = [float(f) for f in os.listdir(self.log_base_path) if f[0] != '.']
        params_files = [float(f.split('.')[0]) for f in os.listdir(self.params_base_path) if f[0] != '.']
        if len(model_major) == 0 or len(log_files) == 0 or len(params_files) == 0:
            serial = 0
        else:
            serial = max([max(model_major), max(log_files), max(params_files)])
        return int(serial)

    def gen_paths(self, serial=None):      
        if serial is None:
            serial = self.get_max_serial()

        m = os.path.join(self.model_base_path, str(serial + 1))
        l = os.path.join(self.log_base_path, str(serial + 1))
        p = os.path.join(self.params_base_path, str(serial + 1))
        return m, l, p

    def get_paths(self, serial=None):
        """
        Example of serial: '1-1'
        """
        if serial is None:
            major_num, minor_num = self.list_model_files()
            major = max(major_num)
            minor = max(minor_num)
        else:
            major = serial.split('-')[0]
            minor = serial.split('-')[1]

        m = os.path.join(self.model_base_path, str(major)+'-'+str(minor))
        p = os.path.join(self.params_base_path, str(major))
        o = os.path.join(self.out_base_path, str(major))
        return m, p, o
