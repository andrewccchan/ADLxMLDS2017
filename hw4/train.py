#!/usr/bin/env python3
import tensorflow as tf 
import numpy as np
import scipy.misc
import random
import os
import util
from util import get_args
from network import GAN
from tensorflow.contrib.data import Dataset, Iterator
import pdb

def train_val_split(x, y, test_ratio):
    assert(len(x) == len(y))
    x = np.asarray(x)
    y = np.asarray(y)

    data_len = len(x)
    idx = list(range(data_len))
    test_len = int(data_len * test_ratio)
    train_len = data_len - test_len

    indices = np.random.permutation(data_len)
    train_idx = indices[:train_len]
    test_idx = indices[train_len:]

    return x[train_idx,:], y[train_idx,:], x[test_idx,:], y[test_idx,:]

    
def proc_batch_data(data, train_imgs, train_caps, noise_dim):
    correct_images = data[0]
    captions = data[1]
    # TODO: improve the selection of wrong images
    wrong_images = np.zeros(correct_images.shape)
    idx = np.random.choice(train_imgs.shape[0], wrong_images.shape[0])
    wrong_images = train_imgs[idx]
    wrong_caps = train_caps[idx]
    
    z_noise = np.random.uniform(-1, 1, [correct_images.shape[0], noise_dim])

    return correct_images, wrong_images, captions, wrong_caps, z_noise

import os
def train():
    args = get_args()

    # model settings
    # TODO
    batch_size = 64
    n_epoch = 400
    train_len = 200
    test_len = 20
    noise_dim = 120
    network_params = dict(
        batch_size = batch_size,
        img_size = [96, 96],
        cap_dim = 2400,
        cap_emb_dim = 60,
        noise_dim = noise_dim,
        gen_num_filter = 64,
        dis_num_filter = 64,
        learning_rate = 0.0002,
        training = True
    )

    model = GAN(network_params)
    
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

    model_name = 'large_pic_low_dim'

    # Init. tf session
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    log_file = open(model_name, 'w')    
    # Load data
    img_list, cap_list = util.load_data('./data', True, True)

    # Split train/test sets
    train_imgs, train_labels, val_imgs, val_labels = train_val_split(img_list, cap_list, 0.0)

    # tf dataset
    train_data = Dataset.from_tensor_slices((train_imgs, train_labels)).batch(batch_size).shuffle(train_imgs.shape[0])
    val_data = Dataset.from_tensor_slices((val_imgs, val_labels)).batch(batch_size).shuffle(val_imgs.shape[0])

    handle = tf.placeholder(tf.string, shape=[])
    iterator = Iterator.from_string_handle(
        handle, train_data.output_types, train_data.output_shapes)
    next_batch = iterator.get_next()

    train_iterator = train_data.make_initializable_iterator()
    val_interator = val_data.make_initializable_iterator()

    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_interator.string_handle())

    # training loop
    sess.run(train_iterator.initializer)
    for epi in range(n_epoch):
        dis_loss_list = []
        gen_loss_list = []
        # training
        while True:
            try:
                data = sess.run(next_batch, feed_dict={handle: train_handle})
                correct_images, wrong_images, correct_captions, wrong_captions, z_noise = proc_batch_data(data, train_imgs, train_labels, noise_dim)
                # update discriminator
                feed_dict = {
                    model.correct_img: correct_images,
                    model.wrong_img: wrong_images,
                    model.correct_cap: correct_captions,
                    model.wrong_cap: wrong_captions,
                    model.noise: z_noise
                }
                _, dis_loss = sess.run([model.dis_optimizer, model.dis_loss], feed_dict=feed_dict)

                # update generator
                # new noise
                z_noise = np.random.uniform(-1, 1, [correct_images.shape[0], noise_dim])
                feed_dict = {
                    model.correct_img: correct_images,
                    model.wrong_img: wrong_images,
                    model.correct_cap: correct_captions,
                    model.wrong_cap: wrong_captions,
                    model.noise: z_noise
                }
                _, gen_loss, gen_img = sess.run([model.gen_optimizer, model.gen_loss, model.gen_img], feed_dict=feed_dict)

                #  # second update
                # _, gen_loss, gen_img = sess.run([model.gen_optimizer, model.gen_loss, model.gen_img], feed_dict=feed_dict)

                dis_loss_list.append(dis_loss)
                gen_loss_list.append(gen_loss)

            except tf.errors.OutOfRangeError:
                sess.run(train_iterator.initializer)
                break

        # reporting
        msg = 'Epoch {}, d_loss = {}, g_loss = {}'.format(epi, np.mean(dis_loss_list), np.mean(gen_loss_list))
        log_file.write(msg + '\n')
        print(msg) 
        # save image
        scipy.misc.imsave('img/gen_img.jpg', gen_img[0])
        
        if epi % 5 == 4:
            saver.save(sess, os.path.join('./model', model_name), global_step=epi)

if __name__ == '__main__':
    train()
