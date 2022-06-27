import numpy as np
import re
from utils.latin_bert.latin_bert import LatinBERT
from funcy import flatten
import torch

def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range") 


def create_bert_vectors_for_batch(word_pos_id,bert_embedding_table):
    word_pos_id = word_pos_id
    dim = bert_embedding_table[0].shape[0]
    sen_length = len(word_pos_id[0])
    result = torch.zeros(len(word_pos_id),sen_length,dim)#dtype="double")
    c= 0 
    for sentence in word_pos_id:
        c1 = 0 
        for token in sentence:
            result[c][c1] = torch.from_numpy(bert_embedding_table[token])
            c1 += 1
        c +=1
    return result.to('cuda:0')

def load_bert_dict_from_conllu(skt_conllu_paths, use_aug):
    print("CONLLU PATHS FOR BERT",skt_conllu_paths)
    bert_dict = {}
    bertPath= "/mnt/code/latin-bert/models/latin_bert/"
    tokenizerPath= "/mnt/code/latin-bert/models/subword_tokenizer_latin/latin.subword.encoder"
    bert=LatinBERT(tokenizerPath=tokenizerPath, bertPath=bertPath)
    text = ""
    sent_id = ""
    sent_ids = []
    sentences = []
    for current_path in skt_conllu_paths:
        current_file = open(current_path,"r")
        for line in current_file:
            if "sent_id" in line:
                sent_id = line.replace("# sent_id = ","").strip()
            if "text =" in line:
                text = line.replace("# text = ","").strip()
                text = re.sub(r"([.,;:?!])",r" \1",text)
                text = text.lower()
            if re.search("^$",line):
                if re.search("[a-zA-Z]",text) and len(text) > 1:
                    sent_ids.append(sent_id)
                    sentences.append(text)
    bert_sents = bert.get_berts(sentences)
    word_vectors = []
    word_ids = []

    for sentence,sent_id,bert_vectors in zip(sentences,sent_ids,bert_sents):
        sentence = re.sub(" +"," ",sentence)    
        for idx in range(0,len(sentence.split(" "))):
            if idx < len(bert_vectors): # seems that in rare cases the lengths of both don't match up, so we got to catch this.
                current_vector = bert_vectors[idx]
                current_id = sent_id + ":" + str(idx)
                bert_dict[current_id] = current_vector
    return bert_dict, len(list(bert_dict.values())[0])

