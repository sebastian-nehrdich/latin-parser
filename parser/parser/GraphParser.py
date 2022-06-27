from __future__ import print_function
import sys
from os import path, makedirs

sys.path.append(".")
sys.path.append("..")

import argparse
from copy import deepcopy
import json
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from utils.io_ import seeds, Writer, get_logger, prepare_data, rearrange_splits
from utils.models.parsing_gating import BiAffine_Parser_Gated
from utils import load_word_embeddings,load_bert_embeddings#,load_fasttext_embeddings
from utils.tasks import parse
import time
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
import uuid
import pdb
uid = uuid.uuid4().hex[:6]

logger = get_logger('GraphParser')

def read_arguments():
    args_ = argparse.ArgumentParser(description='Sovling GraphParser')
    args_.add_argument('--domain', help='domain/language: ved for Vedic, skt for Sanskrit', required=True)
    args_.add_argument('--rnn_mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn',
                       required=True)
    args_.add_argument('--gating',action='store_true', help='use gated mechanism')
    args_.add_argument('--num_gates', type=int, default=0, help='number of gates for gating mechanism')
    args_.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_.add_argument('--arc_tag_space', type=int, default=128, help='Dimension of tag space')
    args_.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    args_.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_.add_argument('--kernel_size', type=int, default=3, help='Size of Kernel for CNN')
    args_.add_argument('--use_aug', action='store_true', help='use augmented sentences.')
    args_.add_argument('--use_bert', action='store_true', help='use bert sentence embeddings.')
    args_.add_argument('--bert_outside_gpu', action='store_true', help='store BERT sentence embeddings outside of GPU between batches to make training on smaller GPUs possible.')
    args_.add_argument('--use_meaning', action='store_true', help='use meaning embeddings.')
    args_.add_argument('--use_w2v', action='store_true', help='use word2vec sentence embeddings.')
    args_.add_argument('--use_pos_type', action='store_true', help='use part-of-speech type embedding.')
    args_.add_argument('--use_case', action='store_true', help='use case embedding.')    
    args_.add_argument('--use_number', action='store_true', help='use number embedding.')    
    args_.add_argument('--use_gender', action='store_true', help='use gender embedding.')
    args_.add_argument('--use_verb_form', action='store_true', help='use verb form embedding.')
    args_.add_argument('--use_full_pos', action='store_true', help='use full pos embedding.')
    args_.add_argument('--use_class', action='store_true', help='use text class embedding.')
    args_.add_argument('--use_punc', action='store_true', help='use punctuation embedding.')        
    args_.add_argument('--use_char', action='store_true', help='use character embedding and CNN.')
    args_.add_argument('--word_dim', type=int, default=300, help='Dimension of word embeddings')
    args_.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_.add_argument('--initializer', choices=['xavier'], help='initialize model parameters')
    args_.add_argument('--opt', choices=['adam', 'sgd'], help='optimization algorithm')
    args_.add_argument('--momentum', type=float, default=0.9, help='momentum of optimizer')
    args_.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.9], help='betas of optimizer')
    args_.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args_.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam')
    args_.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    args_.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_.add_argument('--arc_decode', choices=['mst', 'greedy'], help='arc decoding algorithm', required=True)
    args_.add_argument('--unk_replace', type=float, default=0.,
                       help='The rate to replace a singleton word with UNK')
    args_.add_argument('--punct_set', nargs='+', type=str, help='List of punctuations')
    args_.add_argument('--word_embedding', choices=['random', 'glove', 'fasttext', 'word2vec'],
                       help='Embedding for words')
    args_.add_argument('--word_path', help='path for word embedding dict - in case word_embedding is not random')
    args_.add_argument('--freeze_word_embeddings', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_.add_argument('--freeze_sequence_taggers', action='store_true', help='frozen the BiLSTMs of the pre-trained taggers.')
    args_.add_argument('--char_embedding', choices=['random','hellwig'], help='Embedding for characters',
                       required=True)
    args_.add_argument('--pos_embedding', choices=['random','one_hot'], help='Embedding for pos',
                       required=True)
    args_.add_argument('--char_path', help='path for character embedding dict')
    args_.add_argument('--pos_path', help='path for pos embedding dict')
    args_.add_argument('--set_num_training_samples', type=int, help='downsampling training set to a fixed number of samples')
    args_.add_argument('--model_path', help='path for saving model file.', required=True)
    args_.add_argument('--load_path', help='path for loading saved source model file.', default=None)
    args_.add_argument('--load_sequence_taggers_paths', nargs='+', help='path for loading saved sequence_tagger saved_models files.', default=None)
    args_.add_argument('--strict',action='store_true', help='if True loaded model state should contin '
                                                            'exactly the same keys as current model')
    args_.add_argument('--eval_mode', action='store_true', help='evaluating model without training it')
    args_.add_argument('--inf_mode', action='store_true', help='infer on dataset')
    args_.add_argument('--inf_path', help='path to inference data file')    
    args_.add_argument('--eval_with_CI', action='store_true', help='evaluating model in constrained inference mode')
    args_.add_argument('--LCM_Path_flag', action='store_true', help='for constrained inference with LCM, flag is used to change path')
    args = args_.parse_args()
    args_dict = {}
    args_dict['domain'] = args.domain
    args_dict['rnn_mode'] = args.rnn_mode
    args_dict['gating'] = args.gating
    args_dict['num_gates'] = args.num_gates
    args_dict['arc_decode'] = args.arc_decode
    args_dict['splits'] = ['train', 'val', 'test']
    args_dict['model_path'] = args.model_path
    if not path.exists(args_dict['model_path']):
        makedirs(args_dict['model_path'])
    args_dict['data_paths'] = {}
    data_path = "data/"
    for split in args_dict['splits']:
        args_dict['data_paths'][split] = data_path + split + '_' + args_dict['domain'] + ".tsv"
    ###################################    

    ###################################
    args_dict['alphabet_data_paths'] = {}
    for split in args_dict['splits']:
        if '_' in args_dict['domain']:
            args_dict['alphabet_data_paths'][split] = data_path + '_' + split + '_' + args_dict['domain'].split('_')[0]
        else:
            args_dict['alphabet_data_paths'][split] = args_dict['data_paths'][split]

    if args.inf_path:
        args_dict['alphabet_data_paths']["inf"] = args.inf_path
        
    args_dict['model_name'] = 'domain_' + args_dict['domain']
    args_dict['full_model_name'] = path.join(args_dict['model_path'],args_dict['model_name'])
    args_dict['load_path'] = args.load_path
    args_dict['load_sequence_taggers_paths'] = args.load_sequence_taggers_paths
    if args_dict['load_sequence_taggers_paths'] is not None:
        args_dict['gating'] = True
        args_dict['num_gates'] = len(args_dict['load_sequence_taggers_paths']) + 1
    else:
        if not args_dict['gating']:
            args_dict['num_gates'] = 0
    args_dict['strict'] = args.strict
    args_dict['num_epochs'] = args.num_epochs
    args_dict['batch_size'] = args.batch_size
    args_dict['hidden_size'] = args.hidden_size
    args_dict['arc_space'] = args.arc_space
    args_dict['arc_tag_space'] = args.arc_tag_space
    args_dict['num_layers'] = args.num_layers
    args_dict['num_filters'] = args.num_filters
    args_dict['kernel_size'] = args.kernel_size
    args_dict['learning_rate'] = args.learning_rate
    args_dict['initializer'] = nn.init.xavier_uniform_ if args.initializer == 'xavier' else None
    args_dict['opt'] = args.opt
    args_dict['momentum'] = args.momentum
    args_dict['betas'] = tuple(args.betas)
    args_dict['epsilon'] = args.epsilon
    args_dict['decay_rate'] = args.decay_rate
    args_dict['clip'] = args.clip
    args_dict['gamma'] = args.gamma
    args_dict['schedule'] = args.schedule
    args_dict['p_rnn'] = tuple(args.p_rnn)
    args_dict['p_in'] = args.p_in
    args_dict['p_out'] = args.p_out
    args_dict['unk_replace'] = args.unk_replace
    args_dict['set_num_training_samples'] = args.set_num_training_samples
    args_dict['punct_set'] = None
    if args.punct_set is not None:
        args_dict['punct_set'] = set(args.punct_set)
        logger.info("punctuations(%d): %s" % (len(args_dict['punct_set']), ' '.join(args_dict['punct_set'])))
    args_dict['freeze_word_embeddings'] = args.freeze_word_embeddings
    args_dict['freeze_sequence_taggers'] = args.freeze_sequence_taggers
    args_dict['word_embedding'] = args.word_embedding
    args_dict['word_path'] = args.word_path
    args_dict['use_char'] = args.use_char
    args_dict['char_embedding'] = args.char_embedding
    args_dict['char_path'] = args.char_path
    args_dict['pos_embedding'] = args.pos_embedding
    args_dict['pos_path'] = args.pos_path
    args_dict['use_pos_type'] = args.use_pos_type
    args_dict['use_case'] = args.use_case
    args_dict['use_bert'] = args.use_bert
    args_dict['bert_outside_gpu'] = args.bert_outside_gpu
    args_dict['use_meaning'] = args.use_meaning
    args_dict['use_aug'] = args.use_aug
    args_dict['use_w2v'] = args.use_w2v
    args_dict['use_number'] = args.use_number
    args_dict['use_gender'] = args.use_gender
    args_dict['use_verb_form'] = args.use_verb_form
    args_dict['use_full_pos'] = args.use_full_pos
    args_dict['use_class'] = args.use_class
    args_dict['use_punc'] = args.use_punc    
    args_dict['pos_dim'] = args.pos_dim
    args_dict['word_dict'] = None
    args_dict['word_dim'] = args.word_dim
    args_dict['bert_dict'] = None
    args_dict['bert_dim'] = None
    if args_dict['use_bert']:
        if not args.inf_mode:
            args_dict['bert_dict'], args_dict['bert_dim'] = load_bert_embeddings.load_bert_dict_from_conllu(list(args_dict['data_paths'].values()), use_aug=args.use_aug)
        else:
            args_dict['bert_dict'], args_dict['bert_dim'] = load_bert_embeddings.load_bert_dict_from_conllu([args.inf_path], use_aug=args.use_aug)

    if args_dict['use_meaning']:
        args_dict['meaning_dict'], args_dict['meaning_dim'] = load_fasttext_embeddings.load_fasttext_dict()

        
    args_dict['char_dict'] = None
    args_dict['char_dim'] = args.char_dim
    if args_dict['char_embedding'] != 'random':
        args_dict['char_dict'], args_dict['char_dim'] = load_word_embeddings.load_embedding_dict(args_dict['char_embedding'],
                                                                                                 args_dict['char_path'])
    args_dict['pos_dict'] = None
    if args_dict['pos_embedding'] != 'random':
        args_dict['pos_dict'], args_dict['pos_dim'] = load_word_embeddings.load_embedding_dict(args_dict['pos_embedding'],
                                                                                                 args_dict['pos_path'])
    args_dict['alphabet_path'] = path.join(args_dict['model_path'], 'alphabets' + '_src_domain_' + args_dict['domain'] + '/')
    args_dict['model_name'] = path.join(args_dict['model_path'], args_dict['model_name'])
    args_dict['eval_mode'] = args.eval_mode
    args_dict['inf_mode'] = args.inf_mode
    args_dict['inf_path'] = args.inf_path
        
    args_dict['eval_with_CI'] = args.eval_with_CI
    args_dict['LCM_Path_flag'] = args.LCM_Path_flag 
    args_dict['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_dict['word_status'] = 'frozen' if args.freeze_word_embeddings else 'fine tune'
    args_dict['char_status'] = 'enabled' if args.use_char else 'disabled'
    args_dict['pos_type_status'] = 'enabled' if args.use_pos_type else 'disabled'
    args_dict['case_status'] = 'enabled' if args.use_case else 'disabled'
    args_dict['number_status'] = 'enabled' if args.use_number else 'disabled'
    args_dict['gender_status'] = 'enabled' if args.use_gender else 'disabled'
    args_dict['verb_form_status'] = 'enabled' if args.use_verb_form else 'disabled'
    args_dict['pos_full_status'] = 'enabled' if args.use_full_pos else 'disabled'
    args_dict['class_status'] = 'enabled' if args.use_class else 'disabled'
    args_dict['punc_status'] = 'enabled' if args.use_punc else 'disabled'

    logger.info("Saving arguments to file")
    save_args(args, args_dict['full_model_name'])
    logger.info("Creating Alphabets")
    alphabet_dict = creating_alphabets(args_dict['alphabet_path'], args_dict['alphabet_data_paths'], args_dict['word_dict'])
    args_dict = {**args_dict, **alphabet_dict}
    ARGS = namedtuple('ARGS', args_dict.keys())
    my_args = ARGS(**args_dict)
    return my_args


def creating_alphabets(alphabet_path, alphabet_data_paths, word_dict):
    train_paths = alphabet_data_paths['train']
    extra_paths = [v for k,v in alphabet_data_paths.items() if k != 'train']
    alphabet_dict = {}
    alphabet_dict['alphabets'] = prepare_data.create_alphabets(alphabet_path,
                                                               train_paths,
                                                               extra_paths=extra_paths,
                                                               max_vocabulary_size=10000000,
                                                               embedd_dict=word_dict,
                                                               lower_case=True)
    for k, v in alphabet_dict['alphabets'].items():
        num_key = 'num_' + k.split('_')[0]
        alphabet_dict[num_key] = v.size()
        logger.info("%s : %d" % (num_key, alphabet_dict[num_key]))
    return alphabet_dict



def construct_embedding_table(alphabet, tokens_dict, dim, token_type='word',use_gpu=True,vocab_size=0):
    if tokens_dict is None:
        return None
    scale = np.sqrt(3.0 / dim)
    if vocab_size == 0:
        vocab_size = alphabet.size()
        
    else:
        if vocab_size < alphabet.size():
            print("ERROR: Actuall VOCAB is larger than hardcoded max size. Please revise! VOCAB SIZE: ", alphabet.size(), dim)
    table = np.empty([vocab_size, dim], dtype=np.float32)
    table[prepare_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, dim]).astype(np.float32)
    oov_tokens = 0
    for token, index in alphabet.items():        
        if token in tokens_dict:
            embedding = tokens_dict[token]
        else:
            embedding = np.random.uniform(-scale, scale, [1, dim]).astype(np.float32)
            oov_tokens += 1
        table[index, :] = embedding
    print('token type : %s, number of oov: %d' % (token_type, oov_tokens))
    if use_gpu:
        table = torch.from_numpy(table)
    return table

def save_args(args, full_model_name):
    arg_path = full_model_name + '.arg.json'
    argparse_dict = vars(args)
    with open(arg_path, 'w') as f:
        json.dump(argparse_dict, f)

def generate_optimizer(args, lr, params):
    '''
    @todo: oh: wir sollten mal RMSProp ausprobieren, wenn das in pytorch existiert. Funktioniert in tf mit RNNs immer am besten.
    '''
    params = filter(lambda param: param.requires_grad, params)
    if args.opt == 'adam':
        return Adam(params, lr=lr, betas=args.betas, weight_decay=args.gamma, eps=args.epsilon)
    elif args.opt == 'sgd':
        return SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.gamma, nesterov=True)
    else:
        raise ValueError('Unknown optimization algorithm: %s' % args.opt)


def save_checkpoint(args, model, optimizer, opt, dev_eval_dict, test_eval_dict, full_model_name):
    path_name = full_model_name + '.pt'
    print('Saving model to: %s' % path_name)
    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'opt': opt,
             'dev_eval_dict': dev_eval_dict,
             'test_eval_dict': test_eval_dict}
    torch.save(state, path_name)

def load_checkpoint(args, model, optimizer, dev_eval_dict, test_eval_dict, start_epoch, load_path, strict=True):
    print('Loading saved model from: %s' % load_path)
    checkpoint = torch.load(load_path, map_location=args.device)
    if checkpoint['opt'] != args.opt:
        raise ValueError('loaded optimizer type is: %s instead of: %s' % (checkpoint['opt'], args.opt))
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)



    if strict:
        generate_optimizer(args, args.learning_rate, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
        dev_eval_dict = checkpoint['dev_eval_dict']
        test_eval_dict = checkpoint['test_eval_dict']
        start_epoch = dev_eval_dict['in_domain']['epoch']
    return model, optimizer, dev_eval_dict, test_eval_dict, start_epoch


def build_model_and_optimizer(args):
    use_gpu_flag = True
    if args.bert_outside_gpu:
        use_gpu_flag = False
    # We hardcode the max. vocab size into the model in order to make inference possible on unseen data
    num_word = 100000 
    num_wordpos = 700000
    num_char = 200
    num_pos = 100
    num_case = 100
    num_number = 10
    num_gender = 10
    num_verb =  100
    num_full =  5000
    num_class =  10
    num_arc = 50
    num_punc = 10    
    word_table = load_word_embeddings.construct_fasttext_table(args.alphabets['word_alphabet'], args.word_dim, num_word, use_gpu_flag)
    char_table = construct_embedding_table(args.alphabets['char_alphabet'], args.char_dict, args.char_dim, token_type='char', vocab_size=num_char)
    pos_type_table = construct_embedding_table(args.alphabets['pos_type_alphabet'], args.pos_dict, args.pos_dim, token_type='pos_type', vocab_size=num_pos)
    case_table = construct_embedding_table(args.alphabets['case_alphabet'], args.pos_dict, args.pos_dim, token_type='case', vocab_size=num_case)
    number_table = construct_embedding_table(args.alphabets['number_alphabet'], args.pos_dict, args.pos_dim, token_type='number', vocab_size=num_number)    
    gender_table = construct_embedding_table(args.alphabets['gender_alphabet'], args.pos_dict, args.pos_dim, token_type='gender', vocab_size=num_gender)
    verb_form_table = construct_embedding_table(args.alphabets['verb_form_alphabet'], args.pos_dict, args.pos_dim, token_type='verb_form', vocab_size=num_verb)
    full_pos_table = construct_embedding_table(args.alphabets['full_pos_alphabet'], args.pos_dict, args.pos_dim, token_type='full_pos', vocab_size=num_full)
    class_table = construct_embedding_table(args.alphabets['class_alphabet'], args.pos_dict, args.pos_dim, token_type='class', vocab_size=num_class)
    punc_table = construct_embedding_table(args.alphabets['punc_alphabet'], args.pos_dict, args.pos_dim, token_type='punc')
    bert_table = construct_embedding_table(args.alphabets['wordpos_id_alphabet'], args.bert_dict, args.bert_dim, token_type='wordpos_id',use_gpu=use_gpu_flag, vocab_size=num_wordpos)


    model = BiAffine_Parser_Gated(args.word_dim, args.bert_dim, num_word, num_wordpos, args.char_dim, num_char, args.use_bert, args.bert_outside_gpu, args.use_w2v, args.use_pos_type, args.use_case, args.use_number,args.use_gender,args.use_verb_form,args.use_full_pos, args.use_char, args.use_class, args.use_punc, args.pos_dim, num_pos, num_case, num_number, num_gender, num_verb, num_full, num_class, num_punc, args.num_filters,


                                  args.kernel_size, args.rnn_mode,
                            args.hidden_size, args.num_layers, args.num_arc,
                            args.arc_space, args.arc_tag_space, args.num_gates,
                                  embedd_word=word_table, embedd_bert=bert_table, embedd_char=char_table, embedd_pos_type=pos_type_table, embedd_case=case_table, embedd_gender=gender_table,embedd_number=number_table,embedd_verb_form=verb_form_table,embedd_full_pos=full_pos_table,embedd_class=class_table,embedd_punc=punc_table,
                            p_in=args.p_in, p_out=args.p_out,  p_rnn=args.p_rnn,
                            biaffine=True, arc_decode=args.arc_decode, initializer=args.initializer)
    print(model)
    optimizer = generate_optimizer(args, args.learning_rate, model.parameters())
    start_epoch = 0
    dev_eval_dict = {'in_domain': initialize_eval_dict()}
    test_eval_dict = {'in_domain': initialize_eval_dict()}
    if args.load_path:
        model, optimizer, dev_eval_dict, test_eval_dict, start_epoch = \
            load_checkpoint(args, model, optimizer,
                            dev_eval_dict, test_eval_dict,
                            start_epoch, args.load_path, strict=args.strict)
    if args.load_sequence_taggers_paths:
        pretrained_dict = {}
        model_dict = model.state_dict()
        for idx, path in enumerate(args.load_sequence_taggers_paths):
            print('Loading saved sequence_tagger from: %s' % path)
            checkpoint = torch.load(path, map_location=args.device)
            for k, v in checkpoint['model_state_dict'].items():
                if 'rnn_encoder.' in k:
                    pretrained_dict['extra_rnn_encoders.' + str(idx) + '.' + k.replace('rnn_encoder.', '')] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if args.freeze_sequence_taggers:
        print('Freezing Classifiers')
        for name, parameter in model.named_parameters():
            if 'extra_rnn_encoders' in name:
                parameter.requires_grad = False
    if args.freeze_word_embeddings:
        model.rnn_encoder.word_embedd.weight.requires_grad = False
        # model.rnn_encoder.char_embedd.weight.requires_grad = False
        # model.rnn_encoder.pos_embedd.weight.requires_grad = False
    device = args.device
    model.to(device)
    return model, optimizer, dev_eval_dict, test_eval_dict, start_epoch, bert_table


def initialize_eval_dict():
    eval_dict = {}
    eval_dict['dp_uas'] = 0.0
    eval_dict['dp_las'] = 0.0
    eval_dict['epoch'] = 0
    eval_dict['dp_ucorrect'] = 0.0
    eval_dict['dp_lcorrect'] = 0.0
    eval_dict['dp_total'] = 0.0
    eval_dict['dp_ucomplete_match'] = 0.0
    eval_dict['dp_lcomplete_match'] = 0.0
    eval_dict['dp_ucorrect_nopunc'] = 0.0
    eval_dict['dp_lcorrect_nopunc'] = 0.0
    eval_dict['dp_total_nopunc'] = 0.0
    eval_dict['dp_ucomplete_match_nopunc'] = 0.0
    eval_dict['dp_lcomplete_match_nopunc'] = 0.0
    eval_dict['dp_root_correct'] = 0.0
    eval_dict['dp_total_root'] = 0.0
    eval_dict['dp_total_inst'] = 0.0
    eval_dict['dp_total'] = 0.0
    eval_dict['dp_total_inst'] = 0.0
    eval_dict['dp_total_nopunc'] = 0.0
    eval_dict['dp_total_root'] = 0.0
    return eval_dict

def in_domain_evaluation(args, datasets, model, optimizer, dev_eval_dict, test_eval_dict, epoch,
                         best_model, best_optimizer, patient, bert_table, bert_outside_gpu):
    # In-domain evaluation
    curr_dev_eval_dict = evaluation(args, datasets['val'], 'val', model, args.domain, epoch, bert_table, bert_outside_gpu, 'current_results')
    is_best_in_domain = dev_eval_dict['in_domain']['dp_lcorrect_nopunc'] <= curr_dev_eval_dict['dp_lcorrect_nopunc'] or \
              (dev_eval_dict['in_domain']['dp_lcorrect_nopunc'] == curr_dev_eval_dict['dp_lcorrect_nopunc'] and
               dev_eval_dict['in_domain']['dp_ucorrect_nopunc'] <= curr_dev_eval_dict['dp_ucorrect_nopunc'])

    if is_best_in_domain:
        for key, value in curr_dev_eval_dict.items():
            dev_eval_dict['in_domain'][key] = value
        curr_test_eval_dict = evaluation(args, datasets['test'], 'test', model, args.domain, epoch, bert_table, bert_outside_gpu, 'current_results')
        for key, value in curr_test_eval_dict.items():
            test_eval_dict['in_domain'][key] = value
        best_model = model #deepcopy(model)
        best_optimizer = optimizer # deepcopy(optimizer)
        patient = 0
    else:
        patient += 1
    if epoch == args.num_epochs:
        # save in-domain checkpoint
        if args.set_num_training_samples is not None:
            splits_to_write = datasets.keys()
        else:
            splits_to_write = ['dev', 'test']
        for split in splits_to_write:
            if split == 'dev':
                eval_dict = dev_eval_dict['in_domain']
            elif split == 'test':
                eval_dict = test_eval_dict['in_domain']
            else:
                eval_dict = None
            write_results(args, datasets[split], args.domain, split, best_model, args.domain, eval_dict, bert_table, bert_outside_gpu)
        print("Saving best model")
        save_checkpoint(args, best_model, best_optimizer, args.opt, dev_eval_dict, test_eval_dict, args.full_model_name)

    print('\n')
    return dev_eval_dict, test_eval_dict, best_model, best_optimizer, patient


def evaluation(args, data, split, model, domain, epoch, bert_table, bert_outside_gpu, str_res='results'):
    # evaluate performance on data
    model.eval()

    eval_dict = initialize_eval_dict()
    eval_dict['epoch'] = epoch
    for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):
        word, word_id,  char, sen_id, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, auto_label, masks, lengths = batch
        bert_vectors = []
        if args.bert_outside_gpu:
            bert_vectors = load_bert_embeddings.create_bert_vectors_for_batch(word_pos_id,bert_table)
        out_arc, out_arc_tag, masks, lengths = model.forward(word, word_pos_id, bert_vectors, char, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, mask=masks, length=lengths)
        # word, char, sen_id, word_pos_id, char, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, auto_label, masks, lengths = batch
        # out_arc, out_arc_tag, masks, lengths = model.forward(word, word_pos_id, char, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, mask=masks, length=lengths)
        heads_pred, arc_tags_pred, _ = model.decode(args.model_path, word, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, full_pos, out_arc, out_arc_tag, mask=masks, length=lengths,
                                                    leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS) # taking full_pos twice here to substitute for input_lemma is _very_ hacky; fix this if possible! 
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        pos_type = pos_type.data.cpu().numpy()
        case = case.data.cpu().numpy()
        gender = gender.data.cpu().numpy()                         
        number = number.data.cpu().numpy()
        verb_form = verb_form.data.cpu().numpy()
        full_pos = full_pos.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        class_tag = class_tag.data.cpu().numpy()
        punc_tag = punc_tag.data.cpu().numpy()
        heads_pred = heads_pred.data.cpu().numpy()
        arc_tags_pred = arc_tags_pred.data.cpu().numpy()

        stats, stats_nopunc, stats_root, num_inst = parse.eval_(word, pos_type, heads_pred, arc_tags_pred, heads,
                                                                arc_tags,
                                                                args.alphabets['word_alphabet'],
                                                                args.alphabets['pos_type_alphabet'],
                                                                         
                                                                lengths, punct_set=args.punct_set, symbolic_root=True)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root
        eval_dict['dp_ucorrect'] += ucorr
        eval_dict['dp_lcorrect'] += lcorr
        eval_dict['dp_total'] += total
        eval_dict['dp_ucomplete_match'] += ucm
        eval_dict['dp_lcomplete_match'] += lcm
        eval_dict['dp_ucorrect_nopunc'] += ucorr_nopunc
        eval_dict['dp_lcorrect_nopunc'] += lcorr_nopunc
        eval_dict['dp_total_nopunc'] += total_nopunc
        eval_dict['dp_ucomplete_match_nopunc'] += ucm_nopunc
        eval_dict['dp_lcomplete_match_nopunc'] += lcm_nopunc
        eval_dict['dp_root_correct'] += corr_root
        eval_dict['dp_total_root'] += total_root
        eval_dict['dp_total_inst'] += num_inst

    eval_dict['dp_uas'] = eval_dict['dp_ucorrect'] * 100 / eval_dict['dp_total']  # considering w. punctuation
    eval_dict['dp_las'] = eval_dict['dp_lcorrect'] * 100 / eval_dict['dp_total']  # considering w. punctuation
    print_results(eval_dict, split, domain, str_res)
    return eval_dict

def constrained_evaluation(args, data, split, model, domain, epoch, str_res='results'):
    # evaluate performance on data
    model.eval()

    eval_dict = initialize_eval_dict()
    eval_dict['epoch'] = epoch
    for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):
        word, word_id,char, sen_id, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, auto_label, masks, lengths = batch
        out_arc, out_arc_tag, masks, lengths = model.forward(word, word_pos_id, char, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, mask=masks, length=lengths)
        heads_pred, arc_tags_pred, _ = model.decode(args.model_path, word, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, full_pos, out_arc, out_arc_tag, mask=masks, length=lengths,
                                                    leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS) # taking full_pos twice here to substitute for input_lemma is _very_ hacky; fix this if possible! 
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        pos_type = pos_type.data.cpu().numpy()
        case = case.data.cpu().numpy()
        gender = gender.data.cpu().numpy()                         
        number = number.data.cpu().numpy()
        verb_form = verb_form.data.cpu().numpy()
        full_pos = full_pos.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        class_tag = class_tag.data.cpu().numpy()
        punc_tag = punc_tag.data.cpu().numpy()
        heads_pred = heads_pred.data.cpu().numpy()
        arc_tags_pred = arc_tags_pred.data.cpu().numpy()

        stats, stats_nopunc, stats_root, num_inst = parse.eval_(word, pos_type, heads_pred, arc_tags_pred, heads,
                                                                arc_tags,
                                                                args.alphabets['word_alphabet'],
                                                                args.alphabets['pos_type_alphabet'],
                                                                         
                                                                lengths, punct_set=args.punct_set, symbolic_root=True)

        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root
        eval_dict['dp_ucorrect'] += ucorr
        eval_dict['dp_lcorrect'] += lcorr
        eval_dict['dp_total'] += total
        eval_dict['dp_ucomplete_match'] += ucm
        eval_dict['dp_lcomplete_match'] += lcm
        eval_dict['dp_ucorrect_nopunc'] += ucorr_nopunc
        eval_dict['dp_lcorrect_nopunc'] += lcorr_nopunc
        eval_dict['dp_total_nopunc'] += total_nopunc
        eval_dict['dp_ucomplete_match_nopunc'] += ucm_nopunc
        eval_dict['dp_lcomplete_match_nopunc'] += lcm_nopunc
        eval_dict['dp_root_correct'] += corr_root
        eval_dict['dp_total_root'] += total_root
        eval_dict['dp_total_inst'] += num_inst

    eval_dict['dp_uas'] = eval_dict['dp_ucorrect'] * 100 / eval_dict['dp_total']  # considering w. punctuation
    eval_dict['dp_las'] = eval_dict['dp_lcorrect'] * 100 / eval_dict['dp_total']  # considering w. punctuation
    print_results(eval_dict, split, domain, str_res)
    return eval_dict

def print_results(eval_dict, split, domain, str_res='results'):
    print('----------------------------------------------------------------------------------------------------------------------------')
    print('Testing model on domain %s' % domain)
    print('--------------- Dependency Parsing - %s ---------------' % split)
    print(
        str_res + ' on ' + split + '  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            eval_dict['dp_ucorrect'], eval_dict['dp_lcorrect'], eval_dict['dp_total'],
            eval_dict['dp_ucorrect'] * 100 / eval_dict['dp_total'],
            eval_dict['dp_lcorrect'] * 100 / eval_dict['dp_total'],
            eval_dict['dp_ucomplete_match'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['dp_lcomplete_match'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['epoch']))
    print(
        str_res + ' on ' + split + '  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            eval_dict['dp_ucorrect_nopunc'], eval_dict['dp_lcorrect_nopunc'], eval_dict['dp_total_nopunc'],
            eval_dict['dp_ucorrect_nopunc'] * 100 / eval_dict['dp_total_nopunc'],
            eval_dict['dp_lcorrect_nopunc'] * 100 / eval_dict['dp_total_nopunc'],
            eval_dict['dp_ucomplete_match_nopunc'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['dp_lcomplete_match_nopunc'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['epoch']))
    print(str_res + ' on ' + split + '  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
        eval_dict['dp_root_correct'], eval_dict['dp_total_root'],
        eval_dict['dp_root_correct'] * 100 / eval_dict['dp_total_root'], eval_dict['epoch']))
    print('\n')
def constrained_write_results(args, data, data_domain, split, model, model_domain, eval_dict):
    str_file = args.full_model_name + '_' + split + '_model_domain_' + model_domain + '_data_domain_' + data_domain
    res_filename = str_file + '_res.txt'
    pred_filename = str_file + '_pred.txt'
    gold_filename = str_file + '_gold.txt'
    if eval_dict is not None:
        # save results dictionary into a file
        with open(res_filename, 'w') as f:
            json.dump(eval_dict, f)

    # save predictions and gold labels into files
    pred_writer = Writer(args.alphabets)
    gold_writer = Writer(args.alphabets)
    pred_writer.start(pred_filename)
    gold_writer.start(gold_filename)
    for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):
        word, word_id, char, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, auto_label, masks, lengths = batch
        out_arc, out_arc_tag, masks, lengths = model.forward(word, sen_id, char, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, mask=masks, length=lengths)
        heads_pred, arc_tags_pred, _ = model.decode(args.model_path, word, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, full_pos, out_arc, out_arc_tag, mask=masks, length=lengths,
                                                    leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS) # taking full_pos twice here to substitute for input_lemma is _very_ hacky; fix this if possible! 
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        pos_type = pos_type.data.cpu().numpy()
        case = case.data.cpu().numpy()
        gender = gender.data.cpu().numpy()                         
        number = number.data.cpu().numpy()
        verb_form = verb_form.data.cpu().numpy()
        full_pos = full_pos.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        class_tag = class_tag.data.cpu().numpy()
        punc_tag = punc_tag.data.cpu().numpy()
        heads_pred = heads_pred.data.cpu().numpy()
        arc_tags_pred = arc_tags_pred.data.cpu().numpy()

        # print('words',word)
        # print('Pos',pos)
        
        # print('heads_pred',heads_pred)
        # print('arc_tags_pred',arc_tags_pred)
        # pdb.set_trace()
        # writing predictions
        pred_writer.write(word, pos_type, case, number, gender, verb_form, full_pos, ner, heads_pred, arc_tags_pred, class_tag, punc_tag, lengths, symbolic_root=True)
        # writing gold labels
        gold_writer.write(word, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, lengths, symbolic_root=True)

    pred_writer.close()
    gold_writer.close()
def write_results(args, data, data_domain, split, model, model_domain, eval_dict, bert_table, bert_outside_gpu):
    str_file = args.full_model_name + '_' + split + '_model_domain_' + model_domain + '_data_domain_' + data_domain
    res_filename = str_file + '_res.txt'
    pred_filename = str_file + '_pred.txt'
    gold_filename = str_file + '_gold.txt'
    if eval_dict is not None:
        # save results dictionary into a file
        with open(res_filename, 'w') as f:
            json.dump(eval_dict, f)

    # save predictions and gold labels into files
    pred_writer = Writer(args.alphabets)
    gold_writer = Writer(args.alphabets)
    pred_writer.start(pred_filename)
    gold_writer.start(gold_filename)
    for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):
        word, word_id, char, sen_id, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, auto_label, masks, lengths = batch
        # pdb.set_trace()
        bert_vectors = None
        if args.bert_outside_gpu:
            bert_vectors = load_bert_embeddings.create_bert_vectors_for_batch(word_pos_id,bert_table)

        out_arc, out_arc_tag, masks, lengths = model.forward(word, word_pos_id, bert_vectors, char, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, mask=masks, length=lengths)
        heads_pred, arc_tags_pred, _ = model.decode(args.model_path, word, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, full_pos, out_arc, out_arc_tag, mask=masks, length=lengths,
                                                    leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS) # taking full_pos twice here to substitute for input_lemma is _very_ hacky; fix this if possible! 
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        word_pos_id = word_pos_id.cpu().numpy()
        sen_id = sen_id.data.cpu().numpy()
        pos_type = pos_type.data.cpu().numpy()
        case = case.data.cpu().numpy()
        gender = gender.data.cpu().numpy()                         
        number = number.data.cpu().numpy()
        verb_form = verb_form.data.cpu().numpy()
        full_pos = full_pos.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        class_tag = class_tag.data.cpu().numpy()
        punc_tag = punc_tag.data.cpu().numpy()        
        heads_pred = heads_pred.data.cpu().numpy()
        arc_tags_pred = arc_tags_pred.data.cpu().numpy()
        # print('words',word)
        # print('Pos',pos)
        
        # print('heads_pred',heads_pred)
        # print('arc_tags_pred',arc_tags_pred)
        # pdb.set_trace()
        # writing predictions
        pred_writer.write(word, word_id,  word_pos_id, sen_id, pos_type, case, number, gender, verb_form, full_pos, heads_pred, arc_tags_pred, class_tag, punc_tag, lengths, symbolic_root=True)
        # writing gold labels
        gold_writer.write(word, word_id, word_pos_id, sen_id, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, lengths, symbolic_root=True)

    pred_writer.close()
    gold_writer.close()

def main():
    logger.info("Reading and creating arguments")
    args = read_arguments()
    logger.info("Reading Data")
    datasets = {}
    for split in args.splits:
        print("Splits are:",split)
        dataset = prepare_data.read_data_to_variable(args.data_paths[split], args.alphabets, args.device, lower_case=True,
                                                     symbolic_root=True,use_aug=args.use_aug)
        datasets[split] = dataset
    if args.set_num_training_samples is not None:
        print('Setting train and dev to %d samples' % args.set_num_training_samples)
        datasets = rearrange_splits.rearranging_splits(datasets, args.set_num_training_samples)
    logger.info("Creating Networks")
    num_data = sum(datasets['train'][1])
    model, optimizer, dev_eval_dict, test_eval_dict, start_epoch, bert_table = build_model_and_optimizer(args)
    best_model = deepcopy(model)
    best_optimizer = deepcopy(optimizer)

    logger.info('Training INFO of in domain %s' % args.domain)
    logger.info('Training on Dependecy Parsing')
    logger.info("train: gamma: %f, batch: %d, clip: %.2f, unk replace: %.2f" % (args.gamma, args.batch_size, args.clip, args.unk_replace))
    logger.info('number of training samples for %s is: %d' % (args.domain, num_data))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (args.p_in, args.p_out, args.p_rnn))
    logger.info("num_epochs: %d" % (args.num_epochs))
    print('\n')

    if not args.eval_mode and not args.inf_mode:
        logger.info("Training")
        num_batches = prepare_data.calc_num_batches(datasets['train'], args.batch_size)
        lr = args.learning_rate
        patient = 0
        decay = 0
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            print('Epoch %d (Training: rnn mode: %s, optimizer: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, decay=%d)): ' % (
                epoch, args.rnn_mode, args.opt, lr, args.epsilon, args.decay_rate, args.schedule, decay))
            model.train()
            total_loss = 0.0
            total_arc_loss = 0.0
            total_arc_tag_loss = 0.0
            total_train_inst = 0.0

            train_iter = prepare_data.iterate_batch_rand_bucket_choosing(
                    datasets['train'], args.batch_size, args.device, unk_replace=args.unk_replace)
            start_time = time.time()
            batch_num = 0
            for batch_num, batch in enumerate(train_iter):
                batch_num = batch_num + 1
                optimizer.zero_grad()
                # compute loss of main task
                word, word_id, char, sen_id, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, auto_label, masks, lengths = batch
                bert_vectors = None
                if args.bert_outside_gpu:
                    bert_vectors = load_bert_embeddings.create_bert_vectors_for_batch(word_pos_id,bert_table)
                out_arc, out_arc_tag, masks, lengths = model.forward(word, word_pos_id, bert_vectors, char, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, mask=masks, length=lengths)
                loss_arc, loss_arc_tag = model.loss(out_arc, out_arc_tag, heads, arc_tags, mask=masks, length=lengths)
                loss = loss_arc + loss_arc_tag
                # pdb.set_trace()

                # update losses
                num_insts = masks.data.sum() - word.size(0)
                total_arc_loss += loss_arc.item() * num_insts
                total_arc_tag_loss += loss_arc_tag.item() * num_insts
                total_loss += loss.item() * num_insts
                total_train_inst += num_insts
                # optimize parameters
                loss.backward()
                clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                time_ave = (time.time() - start_time) / batch_num
                time_left = (num_batches - batch_num) * time_ave

                # update log
                if batch_num % 50 == 0:
                    log_info = 'train: %d/%d, domain: %s, total loss: %.2f, arc_loss: %.2f, arc_tag_loss: %.2f, time left: %.2fs' % \
                               (batch_num, num_batches, args.domain, total_loss / total_train_inst, total_arc_loss / total_train_inst,
                                total_arc_tag_loss / total_train_inst, time_left)
                    sys.stdout.write(log_info)
                    sys.stdout.write('\n')
                    sys.stdout.flush()
            print('\n')
            print('train: %d/%d, domain: %s, total_loss: %.2f, arc_loss: %.2f, arc_tag_loss: %.2f, time: %.2fs' %
                  (batch_num, num_batches, args.domain, total_loss / total_train_inst, total_arc_loss / total_train_inst,
                   total_arc_tag_loss / total_train_inst, time.time() - start_time))

            dev_eval_dict, test_eval_dict, best_model, best_optimizer, patient = in_domain_evaluation(args, datasets, model, optimizer, dev_eval_dict, test_eval_dict, epoch, best_model, best_optimizer, patient, bert_table, args.bert_outside_gpu)
            best_model = model # uncomment this to keep best model
            if patient >= args.schedule:
                lr = args.learning_rate / (1.0 + epoch * args.decay_rate)
                optimizer = generate_optimizer(args, lr, model.parameters())
                print('updated learning rate to %.6f' % lr)
                patient = 0
            print_results(test_eval_dict['in_domain'], 'test', args.domain, 'best_results')
            print('\n')
        for split in datasets.keys():
            if args.eval_with_CI and split not in ['train', 'extra_train', 'extra_dev']:
                print('Currently going on ... ',split)
                eval_dict = constrained_evaluation(args, datasets[split], split, best_model, args.domain, epoch, 'best_results')
                constrained_write_results(args, datasets[split], args.domain, split, model, args.domain, eval_dict)
            else:
                eval_dict = evaluation(args, datasets[split], split, best_model, args.domain, epoch, bert_table, args.bert_outside_gpu, 'best_results')
                write_results(args, datasets[split], args.domain, split, model, args.domain, eval_dict, bert_table, args.bert_outside_gpu)
    # inference code starts here
    elif args.inf_mode:
        model.eval()
        inf_file_path = args.inf_path
        logger.info("Running inference on file " + inf_file_path)                
        data = prepare_data.read_data_to_variable(inf_file_path, args.alphabets, args.device,symbolic_root=True, lower_case=True)
        inf_writer = Writer(args.alphabets)
        inf_writer.start(inf_file_path + "_inf")
        for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):

            word, word_id, char, sen_id, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, heads, arc_tags, class_tag, punc_tag, auto_label, masks, lengths = batch
            bert_vectors = None
            if args.bert_outside_gpu:
                bert_vectors = load_bert_embeddings.create_bert_vectors_for_batch(word_pos_id,bert_table)
            
            # pdb.set_trace()
            out_arc, out_arc_tag, masks, lengths = model.forward(word, word_pos_id, bert_vectors, char, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, mask=masks, length=lengths)            
            heads_pred, arc_tags_pred, _ = model.decode(args.model_path, word, word_pos_id, pos_type, case, number, gender, verb_form, full_pos, class_tag, punc_tag, full_pos, out_arc, out_arc_tag, mask=masks, length=lengths,
                                                    leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS) # taking full_pos twice here to substitute for input_lemma is _very_ hacky; fix this if possible! 
            
            lengths = lengths.cpu().numpy()
            word = word.data.cpu().numpy()
            pos_type = pos_type.data.cpu().numpy()
            case = case.data.cpu().numpy()
            gender = gender.data.cpu().numpy()                         
            number = number.data.cpu().numpy()
            verb_form = verb_form.data.cpu().numpy()
            full_pos = full_pos.data.cpu().numpy()
            heads = heads.data.cpu().numpy()
            arc_tags = arc_tags.data.cpu().numpy()
            class_tag = class_tag.data.cpu().numpy()
            punc_tag = punc_tag.data.cpu().numpy()            
            heads_pred = heads_pred.data.cpu().numpy()
            arc_tags_pred = arc_tags_pred.data.cpu().numpy()
            word_id = word_id.data.cpu().numpy()
            sen_id = sen_id.data.cpu().numpy()
            # writing inference
            inf_writer.write(word, word_id, word_pos_id, sen_id, pos_type, case, number, gender, verb_form, full_pos, heads_pred, arc_tags_pred, class_tag, punc_tag, lengths, symbolic_root=True)


        inf_writer.close()

    else:
        logger.info("Evaluating")
        epoch = start_epoch
        for split in ['train','val','test']:
            if args.eval_with_CI and split != 'train':
                eval_dict = constrained_evaluation(args, datasets[split], split, best_model, args.domain, epoch, 'best_results')
                constrained_write_results(args, datasets[split], args.domain, split, model, args.domain, eval_dict)
            else:
                eval_dict = evaluation(args, datasets[split], split, best_model, args.domain, epoch, 'best_results')
                write_results(args, datasets[split], args.domain, split, model, args.domain, eval_dict)


if __name__ == '__main__':
    main()
