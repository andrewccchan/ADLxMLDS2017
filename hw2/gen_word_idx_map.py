"""
Generate the word_idx_map
"""
#!/usr/bin/env python3
import pickle
import utility
import pandas as pd
from collections import Counter
import numpy as np

import pdb
if __name__ == '__main__':
    train_labels = utility.read_labels('./data/training_label.json')
    test_labels = utility.read_labels('./data/testing_label.json')
    # Combine all captions
    all_vids = pd.concat([train_labels['caption'], test_labels['caption']], axis=0)
    word_cnt = Counter()

    word_id_cnt = 2
    word_idx_map = {'<bos>': 0, '<eos>': 1, '<unk>': 2}
    idx_word_map = {0: '<bos>', 1:'<eos>', 2:'<unk>'}
    for all_caps in all_vids:
        for cap in all_caps:
            cap = '<bos> ' + cap
            cap = cap.replace('.', '')
            cap = cap.replace(',', '')
            cap = cap.replace('"', '')
            cap = cap.replace('\n', '')
            cap = cap.replace('?', '')
            cap = cap.replace('!', '')
            cap = cap.replace('\\', '')
            cap = cap.replace('/', '')
            cap = cap + ' <eos>'
            
            cap_words = cap.lower().split(' ')
            for w in cap_words:
                word_cnt[w] += 1
    vocab = word_cnt.most_common(4000)
    vocab = [word[0] for word in vocab]

    ii = 3
    for v in vocab:
        if v != '<bos>' and v != '<eos>' and v!='<unk>':
            word_idx_map[v] = ii
            idx_word_map[ii] = v
            ii += 1
    word_cnt['<unk>'] = 1000
    bias_init_vector = np.array([1.0*word_cnt[idx_word_map[i]] for i in idx_word_map])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    print(len(vocab))
    with open('word_idx.map', 'wb') as wi:
        pickle.dump(word_idx_map, wi)
    with open('idx_word.map', 'wb') as iw:
        pickle.dump(idx_word_map, iw)
    with open('bias_init', 'wb') as bi:
        pickle.dump(bias_init_vector, bi)
