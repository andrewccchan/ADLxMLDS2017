#!/usr/bin/env python3
# import numpy as np
import tensorflow as tf
import math

def get_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def get_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Only pooling in the feature domain
def max_pool(x, pool_size):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, 1, pool_size, 1], padding='SAME')

def build_cnn_lstm_graph(params):
    """
    CNN as feature extractor for RNN (LSTM) 
    """
    # hyperparameters
    feature_size = params['feature_size']
    num_steps = params['num_steps']
    num_classes = params['num_classes']
    cnn_filter_size = params['cnn_filter_size']
    cnn_filter_num = params['cnn_filter_num'] # list of int (#filter for each layer)
    cnn_layer_num = params['cnn_layer_num']
    cnn_pool_size = params['cnn_pool_size'] # list of int (pooling size)
    fc_layer_size = params['fc_layer_size'] # FC layer sizes (a list of int)
    rnn_state_size = params['rnn_state_size']
    learning_rate = params['learning_rate']

    # rnn_num_steps = params['rnn_num_steps']
    assert(cnn_layer_num == len(cnn_filter_num))

    # Input and output 
    x = tf.placeholder(tf.float32, [None, num_steps, feature_size], \
    name='cnn_input_layer')
    y = tf.placeholder(tf.int32, [None, num_steps], \
    name='rnn_output_layer')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # CNN layers
    # x_4d = tf.reshape(x [None, num_steps, feature_size, 1])
    x_4d = tf.expand_dims(x, 3)
    # Conv. layers
    hs = [x_4d]
    expanded_filter_num = [1] + cnn_filter_num
    for idx in range(cnn_layer_num):
        input_channel = expanded_filter_num[idx]
        output_channel = cnn_filter_num[idx]
        weight = get_weight_variable([cnn_filter_size, cnn_filter_size, input_channel, output_channel])
        bias = get_bias_variable([output_channel])
        hs.append(max_pool(tf.nn.relu(conv2d(hs[-1], weight) + bias), cnn_pool_size[idx]))
    
    conv_output = hs[-1]
    conv_output_dims = conv_output.get_shape().as_list()
    # Expected shape of output (cnn_num_steps, reduced_feature_size, cnn_filter_num[-1])
    print('CNN output. Tensor shape = ', conv_output_dims)

    # Reshape conv. output for FC layers
    fc_dim1 = conv_output_dims[1]
    fc_dim2 = conv_output_dims[2]*conv_output_dims[3]
    fc1_input = tf.reshape(conv_output, [-1, fc_dim1, fc_dim2])
    print('FC input. Tensor shape = ', fc1_input.get_shape().as_list())

    # FC layers
    last_output = tf.reshape(fc1_input, [-1, fc_dim2])
    for idx in range(len(fc_layer_size)):
        print('FC {}. Input tensor shape = {}'.format(idx, last_output.get_shape().as_list()))
        input_dim = last_output.get_shape().as_list()[-1]
        fc_w = get_weight_variable([input_dim, fc_layer_size[idx]])
        fc_b = get_bias_variable([fc_layer_size[idx]])
        # tmp = tf.reshape(hs[-1], [-1, input_dim])
        fc_output = tf.nn.relu(tf.matmul(last_output, fc_w) + fc_b) #NOTE not sure whether to use activation or not
        fc_drop = tf.nn.dropout(fc_output, keep_prob)
        last_output = fc_drop

    print('FC output. Tensor shape = ', last_output.get_shape().as_list())

    rnn_input = tf.reshape(last_output, [-1, num_steps, fc_layer_size[-1]])
    print('RNN input. Tensor shape = ', rnn_input.get_shape().as_list())
    
    # RNN LSTM model
    input_dims = rnn_input.get_shape().as_list()
    batch_size = tf.shape(rnn_input)[0]
    #cell = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)

    #stacked_rnn = []
    #for _ in range(2):
    #    cell = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)
    #    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    #    stacked_rnn.append(cell)
    #cell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
    #cell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)

    cell = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
    rnn_init_state_fw = cell.zero_state(batch_size, tf.float32)
    rnn_init_state_bw = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, rnn_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=rnn_input,
                                initial_state_fw=rnn_init_state_fw, initial_state_bw=rnn_init_state_bw)
    output_fw, output_bw = rnn_outputs
    final_state_fw, final_state_bw = rnn_final_state
    
    

    w_rnn_out_fw = get_weight_variable([rnn_state_size, num_classes])
    b_rnn_out = get_bias_variable([num_classes])
    w_rnn_out_bw = get_weight_variable([rnn_state_size, num_classes])

    # rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_state_size])
    output_fw = tf.reshape(output_fw, [-1, rnn_state_size])
    output_bw = tf.reshape(output_bw, [-1, rnn_state_size])
    logits = tf.matmul(output_fw, w_rnn_out_fw) + tf.matmul(output_bw, w_rnn_out_bw) + b_rnn_out
    tmp_dim1 = input_dims[1]
    logits = tf.reshape(logits, [-1, tmp_dim1, num_classes])

    output_idx = int(math.floor(input_dims[1] / 2)) - 1
    last_output = tf.slice(logits, [0, output_idx, 0], [batch_size, 1, num_classes])
    predictions_one_hot = tf.nn.softmax(tf.squeeze(last_output))
    predictions = tf.argmax(predictions_one_hot, axis=1)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # Add tensors to collection for model saving
    tf.add_to_collection('predictions', predictions)
    tf.add_to_collection('train_step', train_step)
    tf.add_to_collection('total_loss', total_loss)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('keep_prob', keep_prob)
    

    return dict(
        x = x,
        y = y,
        keep_prob = keep_prob,
        init_state_fw = rnn_init_state_fw,
        init_state_bw = rnn_init_state_bw,
        final_state_fw = final_state_fw,
        final_state_bw = final_state_bw,
        total_loss = total_loss,
        train_step = train_step,
        predictions = predictions
    )


def build_lstm_graph(params):
    """
    Building a RNN network with LSTM
    """
    # hyperparameters
    rnn_state_size = params['rnn_state_size']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_steps = params['num_steps']
    # num_layer = 1
    fea_num = params['feature_size']
    learning_rate = params['learning_rate']

    x = tf.placeholder(tf.float32, [None, num_steps, fea_num],\
    name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, num_steps],\
    name='output_placeholder')
    # dummy variable
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    print('RNN input. Tensor shape = ', x.get_shape().as_list())

    batch_size = tf.shape(x)[0]
    # Coding ouput by one-hot encoding
    # y_one_hot = tf.one_hot(y, num_classes, dtype=tf.int32)

    stacked_rnn = []
    for _ in range(2):
        cell = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
        stacked_rnn.append(cell)
    cell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)

    # cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)
    # cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * 2, state_is_tuple=True)
    # cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)
    # cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * 2, state_is_tuple=True)
    rnn_init_state_fw = cell_fw.zero_state(batch_size, tf.float32)
    rnn_init_state_bw = cell_bw.zero_state(batch_size, tf.float32)
    rnn_outputs, rnn_final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=x,
                                initial_state_fw=rnn_init_state_fw, initial_state_bw=rnn_init_state_bw)
    output_fw, output_bw = rnn_outputs
    final_state_fw, final_state_bw = rnn_final_state

    w_rnn_out_fw = get_weight_variable([rnn_state_size, num_classes])
    b_rnn_out = get_bias_variable([num_classes])
    w_rnn_out_bw = get_weight_variable([rnn_state_size, num_classes])

    # rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_state_size])
    output_fw = tf.reshape(output_fw, [-1, rnn_state_size])
    output_bw = tf.reshape(output_bw, [-1, rnn_state_size])
    logits = tf.matmul(output_fw, w_rnn_out_fw) + tf.matmul(output_bw, w_rnn_out_bw) + b_rnn_out
    # tmp_dim1 = input_dims[1]
    logits = tf.reshape(logits, [-1, num_steps, num_classes])

    output_idx = int(math.floor(num_steps / 2)) - 1
    last_output = tf.slice(logits, [0, output_idx, 0], [batch_size, 1, num_classes])
    predictions_one_hot = tf.nn.softmax(tf.squeeze(last_output))
    predictions = tf.argmax(predictions_one_hot, axis=1)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    
    tf.add_to_collection('predictions', predictions)
    tf.add_to_collection('train_step', train_step)
    tf.add_to_collection('total_loss', total_loss)
    # tf.add_to_collection('final_state', final_state)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('keep_prob', keep_prob)

    return dict(
        x = x,
        y = y,
        keep_prob = keep_prob,
        init_state_fw = rnn_init_state_fw,
        init_state_bw = rnn_init_state_bw,
        final_state_fw = final_state_fw,
        final_state_bw = final_state_bw,
        total_loss = total_loss,
        train_step = train_step,
        predictions = predictions
    )

