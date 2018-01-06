#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import scipy.misc
from util import get_args
from network import GAN
import util
import pdb
import pickle
import skimage
import sys

def gen_img(text, cap_dict, model, sess, rep, add_idx, noise):
    vec_list = []
    for t in text:
        fields = t.split(' ')
        if len(fields) == 4:
            cond1 = ' '.join(fields[0:2])
            cond2 = ' '.join(fields[2:4])
            set_cond1 = set()
            set_cond2 = set()
            for key, val in cap_dict.items():
                if cond1 in key:
                    set_cond1.add(key)
                if cond2 in key:
                    set_cond2.add(key)
            candidate = sorted(set_cond1 & set_cond2)
        
        if len(fields) == 2 or len(candidate) == 0:
            cond = ' '.join(fields[0:2])
            set_cond = set()
            for key, val in cap_dict.items():
                if cond in key:
                    set_cond.add(key)
            candidate = sorted(set_cond)
        
        if len(candidate) > 1:
            selection = candidate[1]
        else:
            selection = candidate[0]

        if len(fields) != 2 and len(fields) != 4:
            print('[Warning] invalid input format {}'.format(t))
            selection = 'blue hair red eyes'
        
        if len(candidate) == 0:
            print('[Warning] specified style {} not found in train model'.format(t))
            selection = 'blue hair red eyes'

        vec_list.append(cap_dict[selection][2400:])
    keys = [k for k in list(cap_dict.keys())]
    keys.sort()
    add_keys = [keys[i] for i in add_idx]
    add_vec = [cap_dict[k][2400:] for k in add_keys]
    test_vec = np.concatenate((vec_list, add_vec), axis=0)

    # Generate images
    feed_dict = {
        model.correct_img: np.ones((test_vec.shape[0], 96, 96, 3)),
        model.correct_cap: test_vec,
        model.noise: noise
    }

    gen_imgs = sess.run(model.gen_img, feed_dict=feed_dict)
    for idx, img in enumerate(gen_imgs):
        if idx > 2:
            break
        tmp =  skimage.transform.resize(img, (64, 64))
        scipy.misc.imsave('samples/sample_'+ str(idx+1) + '_' + str(rep) +'.jpg', tmp)

# def gen_img(text, cap_dict, model, sess, rep):
#     vec_list = []
#     for t in text:
#         fields = t.split(' ')
#         candidate = []    
#         if len(fields) == 4:
#             cond1 = ' '.join(fields[0:2])
#             cond2 = ' '.join(fields[2:4])
#             set_cond1 = set()
#             set_cond2 = set()
#             for key, val in cap_dict.items():
#                 if cond1 in key:
#                     set_cond1.add(idx)
#                 if cond2 in key:
#                     set_cond2.add(idx)
#             candidate = sorted(set_cond1 & set_cond2)

#             if len(candidate) > 0:
#                 selection = np.random.choice(candidate, 1, replace=False)

#         if len(fields) == 2 or len(candidate) == 0:
#             cond = ' '.join(fields[0:2])
#             set_cond = set()
#             for key, val in cap_dict.items():
#                 if cond in key:
#                     set_cond.add(idx)
            
#             candidate = sorted(set_cond)
#             selection = np.random.choice(candidate, 1, replace=False)

#         if len(fields) != 2 and len(fields) != 4:
#             print('[Warning] invalid input format {}'.format(t))
#             selection = [0]
        
#         if len(candidate) == 0:
#             print('[Warning] specified style {} not found in train model'.format(t))
#             selection = [0]
        
#         vec_list.append(cap_dict[selection[0]][1][2400:])
    
#     keys = [int(k) for k in list(cap_dict.keys())]
#     keys.sort()
#     all_vecs = np.asarray([cap_dict[i][1][2400:] for i in keys])
#     add_selection = np.random.choice(all_vecs.shape[0], 63).tolist()
#     test_vec = np.concatenate((vec_list, all_vecs[add_selection]), axis=0)
    
#     # Generate images
#     noise = np.random.normal(size=[test_vec.shape[0], 120])        
#     feed_dict = {
#         model.correct_img: np.ones((test_vec.shape[0], 96, 96, 3)),
#         model.correct_cap: test_vec,
#         model.noise: noise
#     }

#     gen_imgs = sess.run(model.gen_img, feed_dict=feed_dict)
#     for idx, img in enumerate(gen_imgs):
#         if idx > 2:
#             break
#         tmp =  skimage.transform.resize(img, (64, 64))
#         scipy.misc.imsave('samples/sample_'+ str(idx+1) + '_' + str(rep) +'.jpg', tmp)

if __name__ == '__main__':
    np.random.seed(0)
    # Read text embeddingn
    with open('data/tags.pkl', 'rb') as it:
        good_tags = pickle.load(it)
    
    # Open test files
    test_sent = []
    with open(sys.argv[1]) as f:
        for l in f:
            l = l.strip()
            if l != '':
                test_sent.append(l.split(',')[1])

    model_path = 'normal_random-329'

    # network settings
    batch_size = 64
    n_epoch = 300
    train_len = 200
    test_len = 20
    noise_dim = 120
    network_params = dict(
        batch_size = batch_size,
        img_size = [96, 96],
        cap_dim = 2400,
        cap_emb_dim = 170,
        noise_dim = noise_dim,
        gen_num_filter = 64,
        dis_num_filter = 64,
        learning_rate = 0.0002,
        training = True
    )

    model = GAN(network_params)
    sess = tf.Session()
    model_saver = tf.train.Saver()
    model_saver.restore(sess, model_path)

    add_idx = np.load('dummy.npy')
    for rep in range(1, 6):
        noise = np.load('noise{}.npy'.format(rep))
        gen_img(test_sent, good_tags, model, sess, rep, add_idx, noise)