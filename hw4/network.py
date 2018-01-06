"""
Definition of GAN network structure
"""
import tensorflow as tf
import pdb

def lrelu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class GAN:
    def __init__(self, params):
        self.params = params
        # Parsing params
        self.batch_size = params['batch_size']
        self.img_size = params['img_size'] # [m, n, c]
        self.cap_dim = params['cap_dim'] # dim. of cpation  vector
        self.cap_emb_dim = params['cap_emb_dim'] # dim. of caption embedding 
        self.noise_dim = params['noise_dim']
        self.gen_num_filter = params['gen_num_filter'] # Ex: 64. Note: a scalar
        self.dis_num_filter = params['dis_num_filter'] # Ex: 64 (scalar) 
        self.learning_rate = params['learning_rate']
        self.training = True if 'training' in params else False        
        self.build_network()
        self.build_optimizers()
    
    def build_network(self):
        # Input place holder
        self.correct_img = tf.placeholder('float32', [None] + self.img_size + [3], name = 'correct_image')
        self.wrong_img = tf.placeholder('float32', [None] + self.img_size + [3], name = 'wrong_image')
        self.correct_cap = tf.placeholder('float32', [None, self.cap_dim], name = 'correct_cap')
        self.wrong_cap = tf.placeholder('float32', [None, self.cap_dim], name = 'wrong_cap')
        self.noise = tf.placeholder('float32', [None, self.noise_dim], name='noise')

        self.batch_size = tf.shape(self.correct_img)[0]
        # Generate an image
        self.gen_img = self.generator(self.noise, self.correct_cap)

        # Discriminate the generated image
        self.dis_correct_image, self.dis_correct_image_acti = self.discriminator(self.correct_img, self.correct_cap)
        self.dis_wrong_image, self.dis_wrong_image_acti = self.discriminator(self.wrong_img, self.correct_cap, reuse=True)
        self.dis_gen_image, self.dis_gen_image_acti = self.discriminator(self.gen_img, self.correct_cap, reuse=True)
        self.dis_wrong_cap, self.dis_wrong_cap_acti = self.discriminator(self.correct_img, self.wrong_cap, reuse=True)

        # Calculate loss
        # Custom generator loss
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_gen_image, labels=tf.ones_like(self.dis_gen_image)))
        self.gen_loss += 100 * tf.reduce_mean(tf.pow(self.dis_correct_image_acti - self.dis_gen_image_acti, 2))
        self.gen_loss += 50 * tf.reduce_mean(tf.abs(self.gen_img - self.correct_img))
       
        self.dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_correct_image, labels=tf.ones_like(self.dis_correct_image) - 0.1))
        self.dis_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_wrong_image, labels=tf.zeros_like(self.dis_wrong_image)))
        self.dis_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_gen_image, labels=tf.zeros_like(self.dis_gen_image)))
        self.dis_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_wrong_cap, labels=tf.zeros_like(self.dis_wrong_cap)))

    def build_optimizers(self):
        all_vars = tf.trainable_variables()
        gen_vars = [v for v in all_vars if 'gen_' in v.name]
        dis_vars = [v for v in all_vars if 'dis_' in v.name]

        self.gen_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.gen_loss, var_list=gen_vars)
        self.dis_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.dis_loss, var_list=dis_vars)
    
    def generator(self, noise, cap):
        with tf.variable_scope('generator'):
            cap_emb_norm = tf.layers.batch_normalization(tf.layers.dense(cap, self.cap_emb_dim, name='gen_cap_lin'), training=self.training, name='gen_cap_norm')
            cap_emb = lrelu(cap_emb_norm)
            z = tf.reshape(tf.concat([cap_emb, noise], 1), [self.batch_size, 1, 1, self.cap_emb_dim + self.noise_dim])
  
            h0 = tf.layers.conv2d_transpose(z, self.gen_num_filter*8, [4, 4], strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='gen_h0')
            h0_ = tf.nn.relu(tf.layers.batch_normalization(h0, training=self.training, name='gen_h0_'))

            h1 = tf.layers.conv2d_transpose(h0_, self.gen_num_filter*4, [4, 4], strides=(3, 3), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='gen_h1')
            h1_ = tf.nn.relu(tf.layers.batch_normalization(h1, training=self.training, name='gen_h1_'))

            h2 = tf.layers.conv2d_transpose(h1_, self.gen_num_filter*2, [4, 4], strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='gen_h2')
            h2_ = tf.nn.relu(tf.layers.batch_normalization(h2, training=self.training, name='gen_h2_'))

            h3 = tf.layers.conv2d_transpose(h2_, self.gen_num_filter, [4, 4], strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='gen_h3')
            h3_ = tf.nn.relu(tf.layers.batch_normalization(h3, training=self.training, name='gen_h3_'))

            h4 = tf.layers.conv2d_transpose(h3_, 3, [4, 4], strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='gen_h4')

            return tf.tanh(h4)

    # def generator(self, noise, cap):
    #     img_size = self.img_size
    #     with tf.variable_scope('generator'):
    #         tmp = [int(i/16) for i in img_size]
    #         cap_emb = lrelu(tf.layers.dense(cap, self.cap_emb_dim, name='gen_cap_emb'))
    #         z = tf.layers.dense(tf.concat([noise, cap_emb], 1), self.gen_num_filter*8*tmp[0]*tmp[1], name='gen_z')
    #         h0 = tf.reshape(z, [-1] + tmp + [self.gen_num_filter*8])
    #         h0_ = tf.nn.relu(tf.layers.batch_normalization(h0, name='gen_h0_'))
            
    #         h1 = tf.layers.conv2d_transpose(h0_, self.gen_num_filter*4, [5, 5], strides=(2, 2), padding='same', name='gen_h1_deconv')
    #         h1_ = tf.nn.relu(tf.layers.batch_normalization(h1, name='gen_h1_'))

    #         h2 = tf.layers.conv2d_transpose(h1_, self.gen_num_filter*2, [5, 5], strides=(2, 2), padding='same', name='gen_h2_deconv')
    #         h2_ = tf.nn.relu(tf.layers.batch_normalization(h2, name='gen_h2_'))

    #         h3 = tf.layers.conv2d_transpose(h2_, self.gen_num_filter, [5, 5], strides=(2, 2), padding='same', name='gen_h3_deconv')
    #         h3_ = tf.nn.relu(tf.layers.batch_normalization(h3, name='gen_h3_'))

    #         h4 = tf.layers.conv2d_transpose(h3_, 3, [5, 5], strides=(2, 2), padding='same', name='gen_h4_deconv')

    #         pdb.set_trace()
    #         gen_output = tf.tanh(h4)/2. + 0.5
    #         return gen_output

    def discriminator(self, img, cap, reuse=False):
         with tf.variable_scope('discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            h0 = tf.layers.conv2d(img, self.dis_num_filter, [4, 4], strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='dis_h0')
            h0_ = lrelu(h0)

            h1 = tf.layers.conv2d(h0_, self.dis_num_filter*2, [4, 4], strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='dis_h1') 
            h1_ = lrelu(tf.layers.batch_normalization(h1, training=self.training, name='dis_h1_norm'))

            h2 = tf.layers.conv2d(h1_, self.dis_num_filter*4, [4, 4], strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='dis_h2') 
            h2_ = lrelu(tf.layers.batch_normalization(h2, training=self.training, name='dis_h2_norm'))

            h3 = tf.layers.conv2d(h2_, self.dis_num_filter*8, [4, 4], strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='dis_h3') 
            h3_ = lrelu(tf.layers.batch_normalization(h3, training=self.training, name='dis_h3_norm'))

            cap_emb = tf.layers.dense(cap, self.cap_emb_dim, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='dis_cap_emb')
            cap_emb_ = lrelu(tf.layers.batch_normalization(cap_emb, training=self.training, name='dis_cap_emb_'))
            cap_emb_ = tf.expand_dims(cap_emb_, 1)
            cap_emb_ = tf.expand_dims(cap_emb_, 2)
            cap_emb_rep = tf.tile(cap_emb_, [1,6,6,1], name='cap_emb_rep')

            h3_concat = tf.concat([h3_, cap_emb_rep], 3, name='h3_concat')

            h4 = tf.layers.conv2d(h3_concat, self.gen_num_filter*8, [4, 4], strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name='dis_h4')

            return h4, h3_

    # def discriminator(self, img, cap, reuse=False):
    #     with tf.variable_scope('discriminator'):
    #         if reuse:
    #             tf.get_variable_scope().reuse_variables()
    #         h0 = lrelu(tf.layers.conv2d(img, self.dis_num_filter, [5, 5], strides=(2, 2), name='dis_h0_conv'))
    #         h1 = tf.layers.batch_normalization(tf.layers.conv2d(h0, self.dis_num_filter*2, [5, 5], strides=(2, 2), padding='same', name='dis_h1_conv'), name='dis_h1_batch')
    #         h1_ = lrelu(h1)
    #         h2 = tf.layers.batch_normalization(tf.layers.conv2d(h1_, self.dis_num_filter*4, [5, 5], strides=(2, 2), padding='same', name='dis_h2_conv'), name='dis_h2_batch')
    #         h2_ = lrelu(h2)
    #         h3 = tf.layers.batch_normalization(tf.layers.conv2d(h2_, self.dis_num_filter*8, [5, 5], strides=(2, 2), padding='same', name='dis_h3_conv'), name='dis_h3_batch')
    #         h3_ = lrelu(h3)

    #         cap_emb = lrelu(tf.layers.dense(cap, self.cap_emb_dim, name='dis_cap_emb'))
    #         cap_emb = tf.expand_dims(cap_emb, 1)
    #         cap_emb = tf.expand_dims(cap_emb, 2)
    #         cap_emb_rep = tf.tile(cap_emb, [1,4,4,1], name='cap_emb_rep')

    #         h3_concat = tf.concat([h3_, cap_emb_rep], 3, name='h3_concat')
    #         # h3_p = lrelu(tf.layers.conv2d(h3_concat, self.gen_num_filter*8, [1, 1], [1,1], name='dis_h3_p'))
    #         h3_p = tf.layers.batch_normalization(tf.layers.conv2d(h3_concat, self.gen_num_filter*8, [1, 1], [1,1], name='dis_h3_p'), name='dis_h4_batch')
    #         h3_p_ = lrelu(h3_p)
    #         h4 = tf.layers.dense(tf.reshape(h3_p_, [self.batch_size, 4*4*self.gen_num_filter*8]), 1, name='dis_h4')
            
    #         return h4, h3_




