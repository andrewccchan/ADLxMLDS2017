import numpy as np
import tensorflow as tf
import random
import pdb

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class ReplayMemory:
    def __init__(self, params):
        self.memory_size = params['memory_size']
        self.observ_size = params['observ_size']
        self.batch_size = params['batch_size']
        self.actions = np.empty(self.memory_size, dtype=np.uint32)
        self.rewards = np.empty(self.memory_size, dtype=np.uint32)
        self.observ = np.empty((self.memory_size, self.observ_size[0], self.observ_size[1]), dtype=np.float32)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = 4
        # number of elements in the memory, <= self.memory_size
        self.count = 0
        # pointer to current element
        self.idx = 0

        # pre-allocation previous and next states
        self.prev_state = np.empty((self.batch_size, self.history_length, self.observ_size[0], self.observ_size[1]), dtype=np.float32)
        self.next_state = np.empty((self.batch_size, self.history_length, self.observ_size[0], self.observ_size[1]), dtype=np.float32)

    def store(self, observ, reward, action, terminal):
        # assert observ.shape == self.observ_size

        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.observ[self.idx, ...] = observ
        self.terminals[self.idx] = terminal
        self.count = max(self.count, self.idx + 1)
        self.idx = (self.idx + 1) % self.memory_size
    
    def get_state(self, index):
        #assert self.idx >= self.history_length - 1
        index = max(index, self.history_length - 1)
        return self.observ[index - (self.history_length - 1) : index + 1, ...]
    
    def sample(self):
        """
        Return a batch of training data
        """
        sample_idx = []
        for cnt in range(self.batch_size):
            while True:
                cand_idx = random.randint(self.history_length-1, self.count-1)
                if cand_idx >= self.idx and cand_idx - self.history_length < self.idx:
                    continue

                if self.terminals[(cand_idx - self.history_length):cand_idx].any():
                    continue
                
                break
            self.next_state[cnt, ...] = self.get_state(cand_idx)
            self.prev_state[cnt, ...] = self.get_state(cand_idx - 1)
            sample_idx.append(cand_idx)

            
        return (np.transpose(self.prev_state, (0, 2, 3, 1)), 
                self.actions[sample_idx], 
                self.rewards[sample_idx], 
                np.transpose(self.next_state, (0, 2, 3, 1)),
                self.terminals[sample_idx])

    def get_memory_size(self):
        return self.memory_size

class History:
    def __init__(self, action_size):
        self.grad_buffer = []
        self.reward_buffer = []
        self.observ_buffer = []
        self.prob_buffer = []
        self.action_size = action_size
        
    
    def store(self, observ, reward, action, action_prob):
        self.observ_buffer.append(observ)
        self.reward_buffer.append(reward)
        y = np.zeros([self.action_size])
        y[action] = 1
        self.grad_buffer.append(np.array(y).astype('float32') - action_prob)
        self.prob_buffer.append(action_prob)
    
    def reset(self):
        self.reward_buffer = []
        self.observ_buffer = []
        self.grad_buffer = []
        self.prob_buffer = []
            
    @property
    def reward_len(self):
        return len(self.reward_buffer)

import os
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
        l = os.path.join(self.log_base_path, str(major))
        return m, p, o, l
