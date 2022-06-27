import os.path
import numpy as np
from .alphabet import Alphabet
from .logger import get_logger
import torch

# Special vocabulary symbols - we always put them at the end.
PAD = "_<PAD>_"
ROOT = "_<ROOT>_"
END = "_<END>_"
_START_VOCAB = [PAD, ROOT, END]

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140]

from .reader import Reader

def create_alphabets(alphabet_directory, train_paths, extra_paths=None, max_vocabulary_size=100000, embedd_dict=None, 
                     min_occurence=1, lower_case=False):
    def expand_vocab(vocab_list,
                     char_alphabet,
                     sent_id_alphabet,
                     word_pos_id_alphabet,
                     pos_type_alphabet,
                     case_alphabet,
                     number_alphabet,                     
                     gender_alphabet,
                     verb_form_alphabet,
                     full_pos_alphabet,
                     arc_alphabet,
                     class_alphabet,
                     punc_alphabet):
        vocab_set = set(vocab_list)
        for data_path in extra_paths:
            with open(data_path, 'r') as file:
                for line in file:
                    if "#" in line:
                        continue
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    if lower_case:
                        tokens[1] = tokens[1].lower()
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = tokens[1]
                    word_id = tokens[2]
                    sent_id = tokens[3]
                    word_pos_id = sent_id + ":" + str(int(tokens[0]) -1)
                    pos_type = tokens[4]
                    case = tokens[5]
                    number = tokens[6]
                    gender = tokens[7]
                    verb_form = tokens[8]
                    full_pos = tokens[9] 
                    arc_tag = tokens[11]
                    class_tag = tokens[12]
                    punc_tag = tokens[13]
                    
                    pos_type_alphabet.add(pos_type)
                    case_alphabet.add(case)
                    word_pos_id_alphabet.add(word_pos_id)
                    gender_alphabet.add(gender)
                    number_alphabet.add(number)
                    sent_id_alphabet.add(sent_id)
                    verb_form_alphabet.add(verb_form)
                    full_pos_alphabet.add(full_pos)                                        
                    arc_alphabet.add(arc_tag)
                    class_alphabet.add(class_tag)
                    punc_alphabet.add(punc_tag)
                    if embedd_dict is not None:
                        if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)
                    else:
                        if word not in vocab_set:
                            vocab_set.add(word)
                            vocab_list.append(word)
        return vocab_list, char_alphabet, sent_id_alphabet, word_pos_id_alphabet, pos_type_alphabet, case_alphabet, number_alphabet, gender_alphabet, verb_form_alphabet, full_pos_alphabet, arc_alphabet, class_alphabet, punc_alphabet

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', default_value=True, singleton=True)
    char_alphabet = Alphabet('character', default_value=True)
    sent_id_alphabet = Alphabet('sent_id', default_value=True)
    word_pos_id_alphabet = Alphabet('wordpos_id', default_value=True)
    pos_type_alphabet = Alphabet('pos_type', default_value=True)
    case_alphabet = Alphabet('case', default_value=True)
    number_alphabet = Alphabet('number', default_value=True)
    gender_alphabet = Alphabet('gender', default_value=True)    
    verb_form_alphabet = Alphabet('verb_form', default_value=True)
    full_pos_alphabet = Alphabet('full_pos', default_value=True)
    arc_alphabet = Alphabet('arc', default_value=True)
    class_alphabet = Alphabet('class', default_value=True)
    punc_alphabet = Alphabet('punc', default_value=True)
    auto_label_alphabet = Alphabet('auto_labeler', default_value=True)
    # NOTE: For the time being, we do NOT want loading of previously stored vocabs; it is creating all kinds of problems for application on unseen material.
    if 1==1: #not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD)
        sent_id_alphabet.add(PAD)
        word_pos_id_alphabet.add(PAD)
        pos_type_alphabet.add(PAD)        
        case_alphabet.add(PAD)
        number_alphabet.add(PAD)
        gender_alphabet.add(PAD)        
        verb_form_alphabet.add(PAD)
        full_pos_alphabet.add(PAD)
        arc_alphabet.add(PAD)
        class_alphabet.add(PAD)
        punc_alphabet.add(PAD)
        auto_label_alphabet.add(PAD)

        char_alphabet.add(ROOT)
        sent_id_alphabet.add(ROOT)
        word_pos_id_alphabet.add(ROOT)
        pos_type_alphabet.add(ROOT)
        case_alphabet.add(ROOT)
        number_alphabet.add(ROOT)
        gender_alphabet.add(ROOT)                
        verb_form_alphabet.add(ROOT)
        full_pos_alphabet.add(ROOT)
        arc_alphabet.add(ROOT)
        class_alphabet.add(ROOT)
        punc_alphabet.add(ROOT)
        auto_label_alphabet.add(ROOT)
        
        char_alphabet.add(END)
        sent_id_alphabet.add(END)
        word_pos_id_alphabet.add(END)
        pos_type_alphabet.add(END)
        case_alphabet.add(END)
        number_alphabet.add(END)
        gender_alphabet.add(END)        
        verb_form_alphabet.add(END)
        full_pos_alphabet.add(END)
        arc_alphabet.add(END)
        class_alphabet.add(END)
        punc_alphabet.add(END)        
        auto_label_alphabet.add(END)

        vocab = dict()
        if isinstance(train_paths, str):
            train_paths = [train_paths]
        for train_path in train_paths:
            with open(train_path, 'r') as file:
                for line in file:
                    if "# " in line:
                        continue
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    if lower_case:
                        tokens[1] = tokens[1].lower()
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = tokens[1]
                    word_id = tokens[2]
                    sent_id = tokens[3]                    
                    word_pos_id = sent_id + ":" + str(int(tokens[0]) -1)
                    pos_type = tokens[4]
                    case = tokens[5]
                    number = tokens[6]
                    gender = tokens[7]
                    verb_form = tokens[8]
                    full_pos = tokens[9] 
                    arc_tag = tokens[11]
                    class_tag = tokens[12]
                    punc_tag = tokens[13]
                    auto_label = "_"
                    if len(tokens) > 14:
                        auto_label = tokens[14]
                    pos_type_alphabet.add(pos_type)
                    sent_id_alphabet.add(sent_id)
                    word_pos_id_alphabet.add(word_pos_id)
                    case_alphabet.add(case)
                    number_alphabet.add(number)                    
                    gender_alphabet.add(gender)
                    verb_form_alphabet.add(verb_form)
                    full_pos_alphabet.add(full_pos)                                        
                    arc_alphabet.add(arc_tag)
                    class_alphabet.add(class_tag)
                    punc_alphabet.add(punc_tag)
                    auto_label_alphabet.add(auto_label)
                    
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurence

        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = [word for word in vocab_list if vocab[word] > min_occurence]
        vocab_list = _START_VOCAB + vocab_list

        if extra_paths is not None:
            vocab_list, char_alphabet, sent_id_alphabet, word_pos_id_alphabet, pos_type_alphabet, case_alphabet, number_alphabet, gender_alphabet, verb_form_alphabet, full_pos_alphabet, arc_alphabet, class_alphabet, punc_alphabet = \
                expand_vocab(vocab_list, char_alphabet, sent_id_alphabet, word_pos_id_alphabet, pos_type_alphabet, case_alphabet, number_alphabet, gender_alphabet, verb_form_alphabet, full_pos_alphabet, arc_alphabet, class_alphabet, punc_alphabet)

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        sent_id_alphabet.save(alphabet_directory)
        word_pos_id_alphabet.save(alphabet_directory)
        pos_type_alphabet.save(alphabet_directory)
        case_alphabet.save(alphabet_directory)
        number_alphabet.save(alphabet_directory)        
        gender_alphabet.save(alphabet_directory)
        verb_form_alphabet.save(alphabet_directory)        
        full_pos_alphabet.save(alphabet_directory)        
        arc_alphabet.save(alphabet_directory)
        class_alphabet.save(alphabet_directory)
        punc_alphabet.save(alphabet_directory)
        auto_label_alphabet.save(alphabet_directory)

    else:
        print('loading saved alphabet from %s' % alphabet_directory)
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        sent_id_alphabet.load(alphabet_directory)
        word_pos_id_alphabet.load(alphabet_directory)
        pos_type_alphabet.load(alphabet_directory)
        case_alphabet.load(alphabet_directory)
        number_alphabet.load(alphabet_directory)        
        gender_alphabet.load(alphabet_directory)
        verb_form_alphabet.load(alphabet_directory)        
        full_pos_alphabet.load(alphabet_directory)        
        arc_alphabet.load(alphabet_directory)
        class_alphabet.load(alphabet_directory)
        punc_alphabet.load(alphabet_directory)
        auto_label_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    sent_id_alphabet.close()
    word_pos_id_alphabet.close()
    pos_type_alphabet.close()
    case_alphabet.close()
    number_alphabet.close()    
    gender_alphabet.close()
    verb_form_alphabet.close()    
    full_pos_alphabet.close()    
    arc_alphabet.close()
    class_alphabet.close()
    punc_alphabet.close()
    auto_label_alphabet.close()
    alphabet_dict = {'word_alphabet': word_alphabet,
                     'char_alphabet': char_alphabet,
                     'sent_id_alphabet': sent_id_alphabet,
                     'wordpos_id_alphabet': word_pos_id_alphabet,
                     'pos_type_alphabet': pos_type_alphabet,
                     'case_alphabet': case_alphabet,
                     'number_alphabet': number_alphabet,                     
                     'gender_alphabet': gender_alphabet,
                     'verb_form_alphabet': verb_form_alphabet,
                     'full_pos_alphabet': full_pos_alphabet,
                     'arc_alphabet': arc_alphabet,
                     'class_alphabet': class_alphabet,
                     'punc_alphabet': punc_alphabet,                     
                     'auto_label_alphabet': auto_label_alphabet}
    return alphabet_dict

def create_alphabets_for_sequence_tagger(alphabet_directory, parser_alphabet_directory, paths):
    logger = get_logger("Create Alphabets")
    print('loading saved alphabet from %s' % parser_alphabet_directory)

    word_alphabet = Alphabet('word', default_value=True, singleton=True)
    char_alphabet = Alphabet('character', default_value=True)
    sent_id_alphabet = Alphabet('sent_id', default_value=True)
    pos_type_alphabet = Alphabet('pos_type', default_value=True)
    case_alphabet = Alphabet('case', default_value=True)
    number_alphabet = Alphabet('number', default_value=True)
    gender_alphabet = Alphabet('gender', default_value=True)    
    verb_form_alphabet = Alphabet('verb_form', default_value=True)
    full_pos_alphabet = Alphabet('full_pos', default_value=True)
    arc_alphabet = Alphabet('arc', default_value=True)
    class_alphabet = Alphabet('class', default_value=True)
    punc_alphabet = Alphabet('punc', default_value=True)    
    auto_label_alphabet = Alphabet('auto_label', default_value=True)
    wordpos_id_alphabet = Alphabet('wordpos_id',default_value=True)
    
    word_alphabet.load(parser_alphabet_directory)
    char_alphabet.load(parser_alphabet_directory)
    sent_id_alphabet.load(parser_alphabet_directory)
    pos_type_alphabet.load(parser_alphabet_directory)
    case_alphabet.load(parser_alphabet_directory)
    gender_alphabet.load(parser_alphabet_directory)
    number_alphabet.load(parser_alphabet_directory)        
    verb_form_alphabet.load(parser_alphabet_directory)        
    full_pos_alphabet.load(parser_alphabet_directory)        
    arc_alphabet.load(parser_alphabet_directory)
    class_alphabet.load(parser_alphabet_directory)
    punc_alphabet.load(parser_alphabet_directory)
    wordpos_id_alphabet.load(parser_alphabet_directory)
    
    try:
        auto_label_alphabet.load(parser_alphabet_directory)
    except:
        print('Creating auto labeler alphabet')
        auto_label_alphabet.add(PAD)
        auto_label_alphabet.add(ROOT)
        auto_label_alphabet.add(END)
        for path in paths:
            with open(path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    tokens = line.split('\t')
                    if len(tokens) > 14:
                        auto_label = tokens[14]
                        auto_label_alphabet.add(auto_label)

    word_alphabet.save(alphabet_directory)
    char_alphabet.save(alphabet_directory)
    sent_id_alphabet.save(alphabet_directory)
    pos_type_alphabet.save(alphabet_directory)
    case_alphabet.save(alphabet_directory)
    gender_alphabet.save(alphabet_directory)
    number_alphabet.save(alphabet_directory)        
    verb_form_alphabet.save(alphabet_directory)        
    full_pos_alphabet.save(alphabet_directory)        
    arc_alphabet.save(alphabet_directory)
    class_alphabet.save(alphabet_directory)
    punc_alphabet.save(alphabet_directory)
    auto_label_alphabet.save(alphabet_directory)
    wordpos_id_alphabet.save(alphabet_directory)
    
    word_alphabet.close()
    char_alphabet.close()
    sent_id_alphabet.close()
    pos_type_alphabet.close()
    case_alphabet.close()
    gender_alphabet.close()
    number_alphabet.close()
    verb_form_alphabet.close()    
    full_pos_alphabet.close()    
    arc_alphabet.close()
    class_alphabet.close()
    punc_alphabet.close()
    auto_label_alphabet.close()
    wordpos_id_alphabet.close()
    
    alphabet_dict = {'word_alphabet': word_alphabet,
                     'char_alphabet': char_alphabet,
                     'sent_id_alphabet': sent_id_alphabet,
                     'pos_type_alphabet': pos_type_alphabet,
                     'case_alphabet': case_alphabet,
                     'gender_alphabet': gender_alphabet,
                     'number_alphabet': number_alphabet,
                     'verb_form_alphabet': verb_form_alphabet,
                     'full_pos_alphabet': full_pos_alphabet,
                     'arc_alphabet': arc_alphabet,
                     'wordpos_id_alphabet': wordpos_id_alphabet,
                     'class_alphabet': class_alphabet,
                     'punc_alphabet': punc_alphabet,
                     'auto_label_alphabet': auto_label_alphabet}

    return alphabet_dict

def read_data(source_path, alphabets, max_size=None,
              lower_case=False, symbolic_root=False, symbolic_end=False, use_aug=True):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % ', '.join(source_path) if type(source_path) is list else source_path)
    counter = 0
    if type(source_path) is not list:
        source_path = [source_path]
    for path in source_path:
        reader = Reader(path, alphabets)
        inst = reader.getNext(lower_case=lower_case, symbolic_root=symbolic_root, symbolic_end=symbolic_end, use_aug=use_aug)
        while inst is not None and (not max_size or counter < max_size):
            if not inst == "SKIP":
                counter += 1
                inst_size = inst.length()
                sent = inst.sentence
                for bucket_id, bucket_size in enumerate(_buckets):
                    if inst_size < bucket_size:
                        data[bucket_id].append([sent.word_ids,
                                                sent.char_id_seqs,
                                                inst.ids['pos_type_alphabet'],
                                                inst.ids['case_alphabet'],
                                                inst.ids['number_alphabet'],
                                                inst.ids['gender_alphabet'],                                                
                                                inst.ids['verb_form_alphabet'],
                                                inst.ids['full_pos_alphabet'],                                            
                                                inst.heads,
                                                inst.ids['arc_alphabet'],
                                                inst.ids['class_alphabet'],
                                                inst.ids['punc_alphabet'],                                                
                                                inst.word_ids,
                                                inst.ids['sent_id_alphabet'],
                                                inst.ids['wordpos_id_alphabet'],
                                                inst.ids['auto_label_alphabet']])
                        max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                        if max_char_length[bucket_id] < max_len:
                            max_char_length[bucket_id] = max_len
                        break
            # else:
            #     print("SKIPPED SENTENCE")
            inst = reader.getNext(lower_case=lower_case, symbolic_root=symbolic_root, symbolic_end=symbolic_end,use_aug=use_aug)
        reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length

def read_data_to_variable(source_path, alphabets, device, max_size=None,
                          lower_case=False, symbolic_root=False, symbolic_end=False,use_aug=True):
    data, max_char_length = read_data(source_path, alphabets,
                                      max_size=max_size, lower_case=lower_case,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end,use_aug=use_aug)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size <= 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        ptid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        csid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        gnid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nmid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        vfid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        fpid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)        
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        aid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        clid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        puid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)        
        woid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        senid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        wordposids_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        mid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        
        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, ptids, csids, nmids, gnids, vfids, fpids, hids, aids, clids, puids, woids, senids, wordposids, mids = inst            
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos type ids
            ptid_inputs[i, :inst_size] = ptids
            ptid_inputs[i, inst_size:] = PAD_ID_TAG
            # case ids
            csid_inputs[i, :inst_size] = csids
            csid_inputs[i, inst_size:] = PAD_ID_TAG
            # gnids
            nmid_inputs[i, :inst_size] = nmids
            nmid_inputs[i, inst_size:] = PAD_ID_TAG
            # ids
            gnid_inputs[i, :inst_size] = gnids
            gnid_inputs[i, inst_size:] = PAD_ID_TAG
            # ids
            vfid_inputs[i, :inst_size] = vfids
            vfid_inputs[i, inst_size:] = PAD_ID_TAG
            # ids
            fpid_inputs[i, :inst_size] = fpids
            fpid_inputs[i, inst_size:] = PAD_ID_TAG            
            # arc ids
            aid_inputs[i, :inst_size] = aids
            aid_inputs[i, inst_size:] = PAD_ID_TAG
            # class ids
            clid_inputs[i, :inst_size] = clids
            clid_inputs[i, inst_size:] = PAD_ID_TAG
            # punctuation ids
            puid_inputs[i, :inst_size] = puids
            puid_inputs[i, inst_size:] = PAD_ID_TAG
            # auto_label ids
            mid_inputs[i, :inst_size] = mids
            mid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # word ids
            woid_inputs[i, :inst_size] = woids
            woid_inputs[i, inst_size:] = PAD_ID_TAG
            # sentence ids
            senid_inputs[i, :inst_size] = senids
            senid_inputs[i, inst_size:] = PAD_ID_TAG
            # word pos ids
            wordposids_inputs[i, :inst_size] = wordposids
            wordposids_inputs[i, inst_size:] = PAD_ID_TAG
            
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if alphabets['word_alphabet'].is_singleton(wid):
                    single[i, j] = 1

        words = torch.LongTensor(wid_inputs)
        chars = torch.LongTensor(cid_inputs)
        pos_type = torch.LongTensor(ptid_inputs)
        case = torch.LongTensor(csid_inputs)
        sen_ids = torch.LongTensor(senid_inputs)
        wordposids = torch.LongTensor(wordposids_inputs)
        gender = torch.LongTensor(gnid_inputs)
        number = torch.LongTensor(nmid_inputs)
        verb_form = torch.LongTensor(vfid_inputs)
        full_pos = torch.LongTensor(fpid_inputs)         
        heads = torch.LongTensor(hid_inputs)
        word_ids = torch.LongTensor(woid_inputs)
        arc = torch.LongTensor(aid_inputs)
        clid = torch.LongTensor(clid_inputs)
        puid = torch.LongTensor(puid_inputs)
        auto_label = torch.LongTensor(mid_inputs)
        masks = torch.FloatTensor(masks)
        single = torch.LongTensor(single)
        lengths = torch.LongTensor(lengths)
        words = words.to(device)
        chars = chars.to(device)
        pos_type = pos_type.to(device)
        case = case.to(device)
        gender = gender.to(device)
        number = number.to(device)
        verb_form = verb_form.to(device)
        full_pos = full_pos.to(device)         
        heads = heads.to(device)
        word_ids = word_ids.to(device)
        sen_ids = sen_ids.to(device)
        wordposids = wordposids.to(device)
        arc = arc.to(device)
        clid = clid.to(device)
        puid = puid.to(device)
        auto_label = auto_label.to(device)
        masks = masks.to(device)
        single = single.to(device)
        lengths = lengths.to(device)

        data_variable.append((words, word_ids, sen_ids, wordposids, chars, pos_type, case, number, gender, verb_form, full_pos, heads, arc, clid, puid, auto_label, masks, single, lengths))

    return data_variable, bucket_sizes

def iterate_batch(data, batch_size, device, unk_replace=0.0, shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size <= 0:
            continue
        words, word_ids, sen_ids, word_pos_ids, chars, pos_type, case, number, gender, verb_form, full_pos, heads, arc, clids, puids, auto_label, masks, single, lengths = data_variable[bucket_id]
        if unk_replace:
            ones = single.data.new(bucket_size, bucket_length).fill_(1)
            noise = masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], word_ids[excerpt], chars[excerpt], sen_ids[excerpt], word_pos_ids[excerpt],  pos_type[excerpt], case[excerpt], number[excerpt], gender[excerpt], verb_form[excerpt], full_pos[excerpt], heads[excerpt], arc[excerpt], clids[excerpt], puids[excerpt], auto_label[excerpt], \
                  masks[excerpt], lengths[excerpt]

def iterate_batch_rand_bucket_choosing(data, batch_size, device, unk_replace=0.0):
    data_variable, bucket_sizes = data
    indices_left = [set(np.arange(bucket_size)) for bucket_size in bucket_sizes]
    while sum(bucket_sizes) > 0:
        non_empty_buckets = [i for i, bucket_size in enumerate(bucket_sizes) if bucket_size > 0]
        bucket_id = np.random.choice(non_empty_buckets)
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]

        words, word_ids, sen_ids, word_pos_ids, chars, pos_type, case, number, gender, verb_form, full_pos, heads, arc, clids, puids, auto_label, masks, single, lengths = data_variable[bucket_id]
        min_batch_size = min(bucket_size, batch_size)
        indices = torch.LongTensor(np.random.choice(list(indices_left[bucket_id]), min_batch_size, replace=False))
        set_indices = set(indices.numpy())
        indices_left[bucket_id] = indices_left[bucket_id].difference(set_indices)
        indices = indices.to(device)
        words = words[indices]
        if unk_replace:
            ones = single.data.new(min_batch_size, bucket_length).fill_(1)
            noise = masks.data.new(min_batch_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single[indices] * noise)
        bucket_sizes = [len(s) for s in indices_left]
        yield words, word_ids[indices], chars[indices], sen_ids[indices], word_pos_ids[indices],  pos_type[indices], case[indices], number[indices], gender[indices], verb_form[indices], full_pos[indices], heads[indices], arc[indices], clids[indices], puids[indices], auto_label[indices], masks[indices], lengths[indices]


def calc_num_batches(data, batch_size):
    _, bucket_sizes = data
    bucket_sizes_mod_batch_size = [int(bucket_size / batch_size) + 1 if bucket_size > 0 else 0 for bucket_size in bucket_sizes]
    num_batches = sum(bucket_sizes_mod_batch_size)
    return num_batches
