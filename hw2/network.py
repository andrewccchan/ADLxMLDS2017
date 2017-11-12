import tensorflow as tf
import pdb

class S2VT:
    """
    An implementation of the S2VT network for video captioning
    """
    def __init__(self, params):
        # Hyper-parameters
        self.state_size = params['state_size']
        self.dict_size = params['dict_size']
        self.batch_size = params['batch_size']
        self.num_frame = params['num_frame']
        self.feat_size = params['feat_size']
        self.sent_len = params['sent_len']
        self.learning_rate = params['learning_rate']
        # max. number of caption for each video
        # self.max_caption_num = params['max_caption_num']

        # Network
        self.v_LSTM_cell = tf.nn.rnn_cell.LSTMCell(self.state_size)
        self.t_LSTM_cell = tf.nn.rnn_cell.LSTMCell(self.state_size)
        # Transform one-hot vector to a feature vector
        self.word_embed = tf.Variable(tf.random_uniform([self.dict_size, self.state_size]\
         ,  -0.1,0.1), name='word_embed') 
        self.t_output_W = tf.Variable(tf.random_uniform([self.state_size,\
         self.dict_size], -0.1,0.1), name='t_output_W')
        self.t_output_b = tf.Variable(tf.zeros(self.dict_size), name='t_output_b')
        
    def build_train_network(self):
        """
        Build the encoder-decoder network for training
        """
        # Inputs
        vid_input = tf.placeholder(tf.float32, [None, self.num_frame, self.feat_size])
        caption_input = tf.placeholder(tf.int32, [None, self.sent_len])
        caption_mask = tf.placeholder(tf.float32, [None, self.sent_len])

        batch_size = tf.shape(vid_input)[0]
        # State variables
        v_LSTM_states = (tf.zeros((batch_size, self.v_LSTM_cell.state_size[0])),
                        tf.zeros((batch_size, self.v_LSTM_cell.state_size[1])))
        t_LSTM_states = (tf.zeros((batch_size, self.t_LSTM_cell.state_size[0])),
                        tf.zeros((batch_size, self.t_LSTM_cell.state_size[1])))
        padding = tf.zeros([batch_size, self.state_size])

        loss = 0.0
        # Encoder network
        # To ensure reuse is False when calling Adam 
        with tf.variable_scope(tf.get_variable_scope()):
            for idx in range(self.num_frame):
                if idx > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope('v_LSTM'):
                    v_output, v_LSTM_states = self.v_LSTM_cell(vid_input[:,idx,:], v_LSTM_states)
                with tf.variable_scope('t_LSTM'):
                    _, t_LSTM_states = self.t_LSTM_cell(tf.concat([padding, v_output], 1), t_LSTM_states)
                
            null_video = tf.zeros([batch_size, self.feat_size])
            for idx in range(self.sent_len):
                tf.get_variable_scope().reuse_variables()
                # pdb.set_trace()            
                # Decoder network
                with tf.variable_scope('v_LSTM'):
                    v_output, v_LSTM_states = self.v_LSTM_cell(null_video, v_LSTM_states)  
                # Lookup word embedding for each word in current time frame
                caption_embed = tf.nn.embedding_lookup(self.word_embed, caption_input[:,idx])
                with tf.variable_scope('t_LSTM'):
                    t_output, t_LSTM_states = self.t_LSTM_cell(tf.concat([v_output, caption_embed], 1), t_LSTM_states)
                logit_output = tf.nn.xw_plus_b(t_output, self.t_output_W, self.t_output_b)
                # Label processing
                caption_onehot = tf.one_hot(caption_input[:,idx], self.dict_size)
                # Calculate loss
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_output, labels=caption_onehot)
                cross_entropy = cross_entropy * caption_mask[:,idx]

                loss += tf.reduce_mean(cross_entropy)
        # Average loss
        # loss = loss / tf.reduce_sum(caption_mask)
        # pdb.set_trace()
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        # train_op = None

        tf.add_to_collection('x', vid_input)
        tf.add_to_collection('y', caption_input)
        tf.add_to_collection('y_mask', caption_mask)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('train_op', train_op)
        
        return dict(
            x = vid_input,
            y = caption_input,
            y_mask = caption_mask,
            loss = loss,
            train_op = train_op
        )

    def build_test_network(self):
        """
        Build the encoder-decoder network for testing
        """
        # Inputs
        vid_input = tf.placeholder(tf.float32, [None, self.num_frame, self.feat_size])
        batch_size = tf.shape(vid_input)[0]
        
        # State variables
        v_LSTM_states = (tf.zeros((batch_size, self.v_LSTM_cell.state_size[0])),
                        tf.zeros((batch_size, self.v_LSTM_cell.state_size[1])))
        t_LSTM_states = (tf.zeros((batch_size, self.t_LSTM_cell.state_size[0])),
                        tf.zeros((batch_size, self.t_LSTM_cell.state_size[1])))
        padding = tf.zeros([batch_size, self.state_size])

        outputs = []        
        loss = 0.0
        # Encoder network
        # vid_input_list = tf.split(vid_input, self.num_frame, 1)
        with tf.variable_scope(tf.get_variable_scope()):
            for idx in range(self.num_frame):
                if idx > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope('v_LSTM'):
                    v_output, v_LSTM_states = self.v_LSTM_cell(vid_input[:,idx,:], v_LSTM_states)
                with tf.variable_scope('t_LSTM'):
                    _, t_LSTM_states = self.t_LSTM_cell(tf.concat([padding, v_output], 1), t_LSTM_states)
           
            null_video = tf.zeros([batch_size, self.feat_size])
            for idx in range(self.sent_len):
                tf.get_variable_scope().reuse_variables()
                if idx == 0:
                    caption_embed = tf.nn.embedding_lookup(self.word_embed, tf.ones([batch_size], dtype=tf.int64))
                # Decoder network
                with tf.variable_scope('v_LSTM'):
                    v_output, v_LSTM_states = self.v_LSTM_cell(null_video, v_LSTM_states)  
                # pdb.set_trace()
                with tf.variable_scope('t_LSTM'):
                    t_output, t_LSTM_states = self.t_LSTM_cell(tf.concat([caption_embed, v_output], 1), t_LSTM_states)
                logit_output = tf.nn.xw_plus_b(t_output, self.t_output_W, self.t_output_b)
                
                # Produce output
                # pdb.set_trace()
                max_prob_index = tf.argmax(logit_output, 1)
                outputs.append(max_prob_index)

                caption_embed = tf.nn.embedding_lookup(self.word_embed, max_prob_index)
                # caption_embed = tf.expand_dims(caption_embed, 0)
        
        return dict(
            x = vid_input,
            outputs = outputs
        )

    # test.py
    # outputs = sess.run(outputs)
    # outputs = zip(outputs)



        
            

