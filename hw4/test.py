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

def test():
    args = get_args()

    input_file = './data/test.npy'
    model_path = '/Volumes/Andrew/model/normal_random-259'

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
    if args.test:
        model_saver = tf.train.Saver()
        model_saver.restore(sess, model_path)
    
    with open('data/id_tag_dict2.pkl', 'rb') as it:
        cap_dict = pickle.load(it)
    
    keys = [int(k) for k in list(cap_dict.keys())]
    keys.sort()
    all_vecs = np.asarray([cap_dict[i][1][2400:] for i in keys])
    
    gen_idx = 23
    for idx in range(5):
        selection = np.random.choice(all_vecs.shape[0], 63).tolist()
        test_vecs = np.concatenate((np.expand_dims(all_vecs[gen_idx], 0), all_vecs[selection]), axis=0)
        noise = np.random.normal(size=[64, noise_dim])        
        feed_dict = {
            model.correct_img: np.ones((64, 96, 96, 3)),
            model.correct_cap: test_vecs,
            model.noise: noise
        }
        gen_img = sess.run(model.gen_img, feed_dict=feed_dict)
        tmp =  skimage.transform.resize(gen_img[0], (64, 64))
        scipy.misc.imsave('img/early'+str(idx)+'.jpg', tmp)

    # debug
    # eval_tensor = [model.debug['z'], model.debug['h0_'], model.debug['h1_'], model.debug['h2_'], model.debug['h3_'], model.debug['h4']]
    # z, h0, h1, h2, h3, h4 = sess.run(eval_tensor, feed_dict=feed_dict)
    # pdb.set_trace()
    


if __name__ == '__main__':
    test()