"""
Utilities for the program
"""
import argparse
import os
import skimage
import skimage.io
import skimage.transform
import pickle
import pdb
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='whether to train the model')
    parser.add_argument('--test', action='store_true', help='whether to test the model')

    return parser.parse_args()

# def load_data(data_base_path, preprocess=True, train=False):
#     img_list = []
#     emb_list = []
    
#     if train:
#         emb_arr = np.load(os.path.join(data_base_path, 'tags_encoding.npy'))
#         idx_list = []
#         with open(os.path.join(data_base_path, 'idx.txt')) as id_file:
#             idx_list = [int(l.strip()) for l in id_file]
        
#         for i, name in enumerate(idx_list):
#             img_file = os.path.join(data_base_path, 'faces', str(name)+'.jpg')
#             img = skimage.io.imread(img_file)
#             img_list.append(img)
#             emb_list.append(emb_arr[i])
#     pdb.set_trace()
#     return img_list, emb_list



def load_data(data_base_path, preprocess=True, train=False):
    # load image
    img_list = []
    img_names = []
    cap_text = []
    
    

    if train:
        # load caption first
        with open(os.path.join(data_base_path, 'id_tag_dict2.pkl'), 'rb') as tp:
            cap_dict = pickle.load(tp)
        
        keys = [int(k) for k in list(cap_dict.keys())]
        keys.sort()
        # load images
        img_files = [os.path.join(data_base_path, 'faces', str(k)+'.jpg') for k in keys]
        for f in img_files:
            img = skimage.io.imread(f)
            if preprocess:
                img = skimage.transform.resize(img, (96, 96))
                img = img.astype('float32')
                img = img - 0.5
                img_list.append(img)
            else:
                img_list.append(img)
        # load tag vectors
        cap_list = [cap_dict[k][1][2400:] for k in keys]
        # cap_text = [cap_dict[k][0] for k in keys]

        # pdb.set_trace()
        
        return img_list, cap_list
    else:
        img_files = os.listdir(os.path.join(data_base_path, 'faces'))
        for f in img_files:
            img = skimage.io.imread(f)
            if preprocess:
                img = skimage.transform.resize(img, (64, 64))
            img_list.append(img)
        return img_list
            

# import pdb
# if __name__ == '__main__':
#     img_list, cap_list = load_data('./data', True, True)
#     pdb.set_trace()
