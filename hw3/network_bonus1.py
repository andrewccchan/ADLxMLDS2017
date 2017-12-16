import numpy as np
import tensorflow as tf
from utility import clipped_error
import pdb

class QNetwork:
    def __init__(self, params, sess):
        self.learning_rate = params['learning_rate']
        self.action_size = params['action_size'] 
        self.do_duel_q = params['do_duel_q']
        self.params = params
        self.sess = sess        
        self.build_train_network()
        self.build_target_network()
        self.build_optimizer()
        self.sess.run(tf.global_variables_initializer())
        self.update_target_network()

    def build_train_network(self):
        self.w = {}
        self.input = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.conv1, self.w['conv1_w'], self.w['conv1_b'] = \
            self.conv2d(self.input, 32, [8, 8], [4, 4], scope='conv1')
        self.conv2, self.w['conv2_w'], self.w['conv2_b'] = \
            self.conv2d(self.conv1, 64, [4, 4], [2, 2], scope='conv2')
        self.conv3, self.w['conv3_w'], self.w['conv3_b'] = \
            self.conv2d(self.conv2, 64, [3, 3], [1, 1], scope='conv3')
        
        shape = self.conv3.get_shape().as_list()
        self.conv3_flat = tf.reshape(self.conv3, [-1, shape[1]*shape[2]*shape[3]])
        
        if self.do_duel_q:
            self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
                self.linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, scope='value_hid')
            self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
                self.linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, scope='adv_hid')
            self.value, self.w['val_w_out'], self.w['val_w_b'] = \
                self.linear(self.value_hid, 1, scope='value_output')
            self.adv, self.w['adv_w_out'], self.w['adv_w_b'] = \
                self.linear(self.adv_hid, self.action_size, scope='adv_out')
            
            self.qval = self.value + (self.adv - tf.reduce_mean(self.adv, reduction_indices=1, keep_dims=True))
        else:
            self.h1, self.w['h1_w'], self.w['h1_b'] = \
                self.linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, scope='h1')
            self.qval, self.w['h2_w'], self.w['h2_b'] = \
                self.linear(self.h1, self.action_size, scope='qval')
        
        self.q_action = tf.argmax(self.qval, 1)

        # Logging
        q_summary = []
        avg_q = tf.reduce_mean(self.qval, 0)
        for idx in range(self.action_size):
            q_summary.append(tf.summary.histogram('q/{}'.format(idx), avg_q[idx]))
        self.q_summary = tf.summary.merge(q_summary, 'q_summary')

    def build_target_network(self):
        self.t_w = {}
        self.target_input = tf.placeholder(tf.float32, [None, 84, 84, 4])
        self.target_conv1, self.t_w['conv1_w'], self.t_w['conv1_b'] = \
            self.conv2d(self.target_input, 32, [8, 8], [4, 4], scope='target_conv1')
        self.target_conv2, self.t_w['conv2_w'], self.t_w['conv2_b'] = \
            self.conv2d(self.target_conv1, 64, [4, 4], [2, 2], scope='target_conv2')
        self.target_conv3, self.t_w['conv3_w'], self.t_w['conv3_b'] = \
            self.conv2d(self.target_conv2, 64, [3, 3], [1, 1], scope='target_conv3')
        
        shape = self.target_conv3.get_shape().as_list()
        self.target_conv3_flat = tf.reshape(self.target_conv3, [-1, shape[1]*shape[2]*shape[3]])
        
        if self.do_duel_q:
            self.t_value_hid, self.t_w['l4_val_w'], self.t_w['l4_val_b'] = \
                self.linear(self.target_conv3_flat, 512, activation_fn=tf.nn.relu, scope='target_value_hid')
            self.t_adv_hid, self.t_w['l4_adv_w'], self.t_w['l4_adv_b'] = \
                self.linear(self.target_conv3_flat, 512, activation_fn=tf.nn.relu, scope='target_adv_hid')
            self.t_value, self.t_w['val_w_out'], self.t_w['val_w_b'] = \
                self.linear(self.t_value_hid, 1, scope='target_value_output')
            self.t_adv, self.t_w['adv_w_out'], self.t_w['adv_w_b'] = \
                self.linear(self.t_adv_hid, self.action_size, scope='target_adv_out')
            
            self.target_qval = self.t_value + (self.t_adv - tf.reduce_mean(self.t_adv, reduction_indices=1, keep_dims=True))
        else:
            self.target_h1, self.t_w['h1_w'], self.t_w['h1_b'] = \
                self.linear(self.target_conv3_flat, 512, activation_fn=tf.nn.relu, scope='target_h1')
            self.target_qval, self.t_w['h2_w'], self.t_w['h2_b'] = \
                self.linear(self.target_h1, self.action_size, scope='target_qval')
        
        self.selection_idx = tf.placeholder('int32', [None, 2], 'selection_idx')
        self.selected_q_val = tf.gather_nd(self.target_qval, self.selection_idx)

        with tf.variable_scope('target_update'):
            self.t_w_input = {}
            self.t_w_update = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_update[name] = self.t_w[name].assign(self.t_w_input[name])
    
    def build_optimizer(self):
        self.target_q_t = tf.placeholder('float32', [None], 'target_q_t')
        self.action = tf.placeholder('int32', [None], 'action')
        self.learning_rate_step = tf.placeholder('int32', None, name='learning_rate_step')
        action_one_hot = tf.one_hot(self.action, self.action_size, name='action_one_hot')
        action_q = tf.reduce_sum(tf.multiply(self.qval, action_one_hot), axis=1, name='action_q')

        self.td_error = self.target_q_t - action_q
        self.loss = tf.reduce_mean(clipped_error(self.td_error), name='loss')
        self.learning_rate_op = tf.maximum(self.params['learning_rate_minimum'],
        tf.train.exponential_decay(
            self.params['learning_rate'],
            self.learning_rate_step,
            self.params['learning_rate_decay_step'],
            self.params['learning_rate_decay'], 
            staircase=True))
        self.train_op = tf.train.RMSPropOptimizer(
            self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

    def update_target_network(self):
        for w_key in self.w.keys():
            self.t_w_update[w_key].eval({self.t_w_input[w_key]: self.w[w_key].eval(session=self.sess)}, session=self.sess)

    def conv2d(self, x, n_filter, kerner_size, stride, activation_fn=tf.nn.relu, scope='conv2d'):
        with tf.variable_scope(scope):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kerner_size[0], kerner_size[1], x.get_shape()[-1], n_filter]
            
            w = tf.get_variable('w', kernel_size, tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(x, w, stride, padding='VALID')

            b = tf.get_variable('b', [n_filter], initializer=tf.constant_initializer(0.0))
            out = conv + b
            
            out = activation_fn(out)

        return out, w, b

    def linear(self, x, n_hid, activation_fn=None, scope='linear'):
        shape = x.get_shape().as_list()
        with tf.variable_scope(scope):
            w = tf.get_variable('w', [shape[1], n_hid], tf.float32,
                tf.random_normal_initializer(stddev=0.02))
            b = tf.get_variable('bias', [n_hid],
                initializer=tf.constant_initializer(0))
            
            out = tf.matmul(x, w) + b
            
            if activation_fn != None:
                return activation_fn(out), w, b
            else:
                return out, w, b


class ValueNetwork:
    def __init__(self, params):
        # parameter parsing
        self.action_size = params['action_size']
        self.learning_rate = params['learning_rate']
        
        # model building
        self.build_network()
        self.build_optimizer()
    
    def build_network(self):
        from keras.models import Sequential
        from keras.layers.convolutional import Convolution2D
        from keras.layers import Dense, Reshape, Flatten        
        
        
        self.model = Sequential()
        self.model.add(Convolution2D(32, (6, 6), activation="relu", padding="same", input_shape=(80, 80, 1), strides=(3, 3), kernel_initializer="he_uniform"))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(Dense(self.action_size, activation='softmax'))
    
    def build_optimizer(self):
        from keras.optimizers import Adam
        self.optimizer = Adam(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)