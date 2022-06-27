import fasttext
import fasttext.util
import torch
import pickle
import numpy as np
from gensim.models import KeyedVectors
import gzip
import io
import os
import re

def calc_mean_vec_for_lower_mapping(embedd_dict):
    lower_counts = {}
    for word in embedd_dict:
        word_lower = word.lower()
        if word_lower not in lower_counts:
            lower_counts[word_lower] = [word]
        else:
            lower_counts[word_lower] = lower_counts[word_lower] + [word]
    # calculating mean vector for all words that have the same mapping after performing lower()
    for word in lower_counts:
        embedd_dict[word] = np.mean([embedd_dict[word_] for word_ in lower_counts[word]])
    return embedd_dict

def load_embedding_dict(embedding, embedding_path, lower_case=False):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimention, caseless
    """
    print("loading embedding: '%s' from %s" % (embedding, embedding_path))
    if lower_case:
        pkl_path = embedding_path + '_lower' + '.pkl'
    else:
        pkl_path = embedding_path + '.pkl'
    if os.path.isfile(pkl_path):
        # load dict and dim from a pickle file
        with open(pkl_path, 'rb') as f:
            embedd_dict, embedd_dim = pickle.load(f)
        print("num dimensions of word embeddings:", embedd_dim)
        return embedd_dict, embedd_dim

    if embedding == 'glove':
        # loading GloVe
        embedd_dict = {}
        word = None
        with io.open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                embedd_dict[word] = np.fromstring(vec, sep=' ')
        embedd_dim = len(embedd_dict[word])
        if lower_case:
            embedd_dict = calc_mean_vec_for_lower_mapping(embedd_dict)
        for k, v in embedd_dict.items():
            if len(v) != embedd_dim:
                print(len(v),embedd_dim)

    elif embedding == 'fasttext':
        # loading GloVe
        embedd_dict = {}
        word = None
        with io.open(embedding_path, 'r', encoding='utf-8') as f:
            # skip first line
            for i, line in enumerate(f):
                if i == 0:
                    continue
                word, vec = line.split(' ', 1)
                embedd_dict[word] = np.fromstring(vec, sep=' ')
        embedd_dim = len(embedd_dict[word])
        if lower_case:
            embedd_dict = calc_mean_vec_for_lower_mapping(embedd_dict)
        for k, v in embedd_dict.items():
            if len(v) != embedd_dim:
                print(len(v),embedd_dim)


    elif embedding == 'one_hot':
        # loading hellwig
        embedd_dict = {}
        word = None
        with io.open(embedding_path, 'r', encoding='utf-8') as f:
            # skip first line
            for i, line in enumerate(f):
                if i == 0:
                    continue
                word, vec = line.split('@', 1)
                embedd_dict[word] = np.fromstring(vec, sep=' ')
        embedd_dim = len(embedd_dict[word])
        if lower_case:
            embedd_dict = calc_mean_vec_for_lower_mapping(embedd_dict)
        for k, v in embedd_dict.items():
            if len(v) != embedd_dim:
                print(len(v),embedd_dim)

    elif embedding == 'word2vec':
        # loading word2vec
        embedd_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        if lower_case:
            embedd_dict = calc_mean_vec_for_lower_mapping(embedd_dict)
        embedd_dim = embedd_dict.vector_size

    else:
        raise ValueError("embedding should choose from [fasttext, glove, word2vec]")

    print("num dimensions of word embeddings:", embedd_dim)
    # save dict and dim to a pickle file
    with open(pkl_path, 'wb') as f:
        pickle.dump([embedd_dict, embedd_dim], f, pickle.HIGHEST_PROTOCOL)
    return embedd_dict, embedd_dim

def construct_fasttext_table(alphabet, word_dim, vocab_size, use_gpu):
    fasttext.util.download_model('la', if_exists='ignore')
    ft = fasttext.load_model("cc.la.300.bin")
    table = np.empty([vocab_size, word_dim], dtype=np.float32)
    for token, index in alphabet.items():
        embedding = ft.get_word_vector(token)
        table[index, :] = embedding
    table = torch.from_numpy(table)
    return table
