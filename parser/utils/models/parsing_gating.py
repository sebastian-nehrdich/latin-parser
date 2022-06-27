import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
from ..nn import Embedding
from ..nn import BiAAttention, BiLinear
from utils.tasks import parse
from ..nn import utils
import pdb


class BiAffine_Parser_Gated(nn.Module):
    def __init__(self,
                 word_dim, bert_dim, num_words, num_bert, char_dim, num_chars, use_bert, bert_outside_gpu, use_w2v, use_pos_type, use_case, use_number, use_gender, use_verb_form, use_full_pos, use_char, use_class, use_punc, pos_dim, num_pos, num_case, num_number, num_gender, num_verb, num_full, num_class, num_punc,  num_filters,
                 kernel_size, rnn_mode, hidden_size, num_layers, num_arcs, arc_space, arc_tag_space, num_gates,
                 embedd_word=None, embedd_bert=None, embedd_char=None, embedd_pos_type=None, embedd_case=None, embedd_gender=None, embedd_number=None, embedd_verb_form=None, embedd_full_pos=None, embedd_class=None, embedd_punc=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33),
                 biaffine=True, arc_decode='mst', initializer=None):
        super(BiAffine_Parser_Gated, self).__init__()
        self.num_gates = num_gates
        self.rnn_encoder = BiRecurrentConv_Encoder(word_dim, bert_dim, num_words, num_bert, char_dim, num_chars, use_bert, bert_outside_gpu, use_w2v, use_pos_type, use_case, use_number, use_gender, use_verb_form, use_full_pos, use_char, use_class, use_punc, pos_dim, num_pos, num_case, num_number, num_gender, num_verb, num_full, num_class, num_punc, num_filters,
                                                   kernel_size, rnn_mode, hidden_size,
                                                   num_layers,
                                                   embedd_word=embedd_word,
                                                   embedd_bert=embedd_bert,
                                                   embedd_char=embedd_char,
                                                   embedd_pos_type=embedd_pos_type,
                                                   embedd_case=embedd_case,
                                                   embedd_gender=embedd_gender,
                                                   embedd_number=embedd_number,
                                                   embedd_verb_form=embedd_verb_form,
                                                   embedd_full_pos=embedd_full_pos,
                                                   embedd_class=embedd_class,
                                                   embedd_punc=embedd_punc,
                                                   p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)
        if self.num_gates >= 2:
            self.extra_rnn_encoders = nn.ModuleDict([[str(i), BiRecurrentConv_Encoder(word_dim, bert_dim, num_words, num_bert, char_dim, num_chars, use_bert, use_w2v, use_pos_type, use_case, use_number, use_gender, use_verb_form, use_full_pos, use_char, use_class, use_punc,pos_dim, num_pos, num_case, num_number, num_gender, num_verb, num_full, num_class, num_punc, num_filters,
                                                   kernel_size, rnn_mode, hidden_size,
                                                   num_layers, embedd_word=embedd_word,
                                                   embedd_bert=embedd_bert,                                                                                
                                                   embedd_char=embedd_char, embedd_pos_type=embedd_pos_type,
                                                   embedd_case=embedd_case,
                                                   embedd_gender=embedd_gender,
                                                   embedd_number=embedd_number,
                                                   embedd_class=embedd_class,
                                                   embedd_punc=embedd_punc,
                                                   embedd_verb_form=embedd_verb_form,
                                                                                      embedd_full_pos=embedd_full_pos,
                                                   p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer)] for i in range(num_gates - 1)])
            self.gate = Gating(num_gates, 2 * hidden_size)
        else:
            self.extra_rnn_encoders = None
            self.gate = None
        self.parser = BiAffine_Parser_Decoder(hidden_size, num_arcs, arc_space, arc_tag_space, biaffine, p_out, arc_decode)

    def forward(self, input_word, input_bert, bert_vectors, input_char, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_class, input_punc, mask=None, length=None, hx=None):
        encoder_output, hn, mask, length = self.rnn_encoder(input_word, input_bert, bert_vectors, input_char, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_class, input_punc, mask, length, hx)
        if self.num_gates >= 2:
            len_extra_encoders = len(self.extra_rnn_encoders.keys())
            extra_enconder_outputs = [self.extra_rnn_encoders[str(i)](input_word, input_bert, input_char, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_class, input_punc, mask, length, hx)[0] for i in range(len_extra_encoders)]
            rnns_output = self.gate(tuple([encoder_output] + extra_enconder_outputs))
        else:
            rnns_output = encoder_output
        out_arc, out_arc_tag = self.parser(rnns_output, mask)
        return out_arc, out_arc_tag, mask, length

    def loss(self, out_arc, out_arc_tag, heads, arc_tags, mask=None, length=None):
        # out_arc shape [batch_size, length, length]
        # out_arc_tag shape [batch_size, length, arc_tag_space]
        loss_arc, loss_arc_tag = self.parser.loss(out_arc, out_arc_tag, heads, arc_tags, mask, length)
        return loss_arc, loss_arc_tag

    def decode(self,model_path, input_word, input_bert, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_class, input_punc, input_lemma, out_arc, out_arc_tag, bert_vectors=None, mask=None, length=None, leading_symbolic=0):
        
        heads_pred, arc_tags_pred, scores = self.parser.unconstrained_decode_mst(model_path, input_word, input_bert, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_lemma, out_arc, out_arc_tag, mask, length, leading_symbolic)
        # heads_pred, arc_tags_pred, scores = self.parser.decode(out_arc, out_arc_tag, mask, length, leading_symbolic)
        return heads_pred, arc_tags_pred, scores 

    def constrained_decode(self,model_path, input_word, input_bert, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_lemma, out_arc, out_arc_tag, mask=None, length=None, leading_symbolic=0):
        heads_pred, arc_tags_pred, scores = self.parser.constrained_decode_mst(model_path, input_word, input_bert, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_lemma, out_arc, out_arc_tag, mask, length, leading_symbolic)
        # heads_pred, arc_tags_pred, scores = self.parser.decode(out_arc, out_arc_tag, mask, length, leading_symbolic)
        return heads_pred, arc_tags_pred, scores 

    def pre_loss(self, out_arc, out_arc_tag, heads, arc_tags, mask=None, length=None, use_log=True, temperature=1.0):
        out_arc, out_arc_tag = self.parser.pre_loss(out_arc, out_arc_tag, heads, arc_tags, mask, length, use_log, temperature)
        return out_arc, out_arc_tag

class BiAffine_Parser_Decoder(nn.Module):
    def __init__(self, hidden_size, num_arcs, arc_space, arc_tag_space, biaffine, p_out, arc_decode):
        super(BiAffine_Parser_Decoder, self).__init__()
        self.num_arcs = num_arcs
        self.arc_space = arc_space
        self.arc_tag_space = arc_tag_space
        self.out_dim = hidden_size * 2 # change this to * 2 when working without shortcut
        self.biaffine = biaffine
        self.p_out = p_out
        self.arc_decode = arc_decode
        self.dropout_out = nn.Dropout(self.p_out)
        self.arc_h = nn.Linear(self.out_dim, self.arc_space)
        self.arc_c = nn.Linear(self.out_dim, self.arc_space)
        self.attention = BiAAttention(self.arc_space, self.arc_space, 1, biaffine=biaffine)
        self.arc_tag_h = nn.Linear(self.out_dim, arc_tag_space)
        self.arc_tag_c = nn.Linear(self.out_dim, arc_tag_space)
        self.bilinear = BiLinear(arc_tag_space, arc_tag_space, num_arcs)

    def forward(self, input, mask):
        # apply dropout for output
        # [batch_size, length, hidden_size] --> [batch_size, hidden_size, length] --> [batch_size, length, hidden_size]
        input = self.dropout_out(input.transpose(1, 2)).transpose(1, 2)

        # output size [batch_size, length, arc_space]
        arc_h = F.elu(self.arc_h(input))
        arc_c = F.elu(self.arc_c(input))

        # output size [batch_size, length, arc_tag_space]
        arc_tag_h = F.elu(self.arc_tag_h(input))
        arc_tag_c = F.elu(self.arc_tag_c(input))

        # apply dropout
        # [batch_size, length, dim] --> [batch_size, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        arc_tag = torch.cat([arc_tag_h, arc_tag_c], dim=1)

        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)
        arc_tag = self.dropout_out(arc_tag.transpose(1, 2)).transpose(1, 2)

        # output from rnn [batch_size, length, tag_space]
        arc_tag_h, arc_tag_c = arc_tag.chunk(2, 1)
        # head shape [batch_size, length, arc_tag_space]
        arc_tag_h = arc_tag_h.contiguous()
        # child shape [batch_size, length, arc_tag_space]
        arc_tag_c = arc_tag_c.contiguous()
        arc = (arc_h, arc_c)
        # [batch_size, length, length]
        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)
        out_arc_tag = (arc_tag_h, arc_tag_c)
        return out_arc, out_arc_tag

    def loss(self, out_arc, out_arc_tag, heads, arc_tags, mask=None, length=None):
        out_arc, out_arc_tag = self.pre_loss(out_arc, out_arc_tag, heads=heads, arc_tags=arc_tags, mask=mask, length=length, use_log=True, temperature=1.0)
        batch_size, max_len = out_arc.size()
        # pdb.set_trace()
        # loss_arc shape [length-1, batch_size]
        ## out_arc.size() = [16,8]
        out_arc = out_arc.t()
        # loss_arc_tag shape [length-1, batch_size]
        ## out_arc_tag.size() = [16,8]
        out_arc_tag = out_arc_tag.t()
        # number of valid positions which contribute to loss (remove the symbolic head for each sentence).
        ## In mask all valid positions are 1's and others are 0.
        num = mask.sum() - batch_size if mask is not None else float(max_len) * batch_size
        dp_loss = -out_arc.sum() / num, -out_arc_tag.sum() / num
        return dp_loss

    def decode(self, out_arc, out_arc_tag, mask, length, leading_symbolic):
        if self.arc_decode == 'mst':
            heads, arc_tags, scores = self.decode_mst(out_arc, out_arc_tag, mask, length, leading_symbolic)
        else: #self.arc_decode == 'greedy'
            heads, arc_tags, scores = self.decode_greedy(out_arc, out_arc_tag, mask, leading_symbolic)
        return heads, arc_tags, scores

    # '''
    # @bug: parameter von model_path bis input_lemma nicht verwendet
    # '''
    def unconstrained_decode_mst(self,model_path, input_word, input_bert, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_lemma, out_arc, out_arc_tag, mask, length, leading_symbolic):
        loss_arc, loss_arc_tag = self.pre_loss(out_arc, out_arc_tag, heads=None, arc_tags=None, mask=mask, length=length, use_log=True, temperature=1.0)
        batch_size, max_len, _ = loss_arc.size()
        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch_size)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()
        # energy shape [batch_size, num_arcs, length, length]
        raw_energy = loss_arc.unsqueeze(1) + loss_arc_tag
        # pdb.set_trace()
        energy = torch.exp(raw_energy)
        # with open('/home/jivnesh/Documents/DCST/energy.npy', 'wb') as f:
        #     np.save(f, energy.data.cpu().numpy())
        constrained_energy = energy.data.cpu().numpy()
        #
        heads, arc_tags, = parse.decode_MST(constrained_energy, length, leading_symbolic=leading_symbolic,
                                           labeled=True)
        heads = from_numpy(heads)
        arc_tags = from_numpy(arc_tags)
        # compute the average score for each tree
        batch_size, max_len = heads.size()
        scores = torch.zeros_like(heads, dtype=energy.dtype, device=energy.device)
        for b_idx in range(batch_size):
            for len_idx in range(max_len):
                scores[b_idx, len_idx] = energy[b_idx, arc_tags[b_idx, len_idx], heads[b_idx, len_idx], len_idx]
        if mask is not None:
            scores = scores.sum(1) / mask.sum(1)
        else:
            scores = scores.sum(1) / max_len
        return heads, arc_tags, scores

    def decode_mst(self, out_arc, out_arc_tag, mask, length, leading_symbolic):
        loss_arc, loss_arc_tag = self.pre_loss(out_arc, out_arc_tag, heads=None, arc_tags=None, mask=mask, length=length, use_log=True, temperature=1.0)
        batch_size, max_len, _ = loss_arc.size()
        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch_size)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()
        # energy shape [batch_size, num_arcs, length, length]
        raw_energy = loss_arc.unsqueeze(1) + loss_arc_tag
        # pdb.set_trace()
        energy = torch.exp(raw_energy)
        # pdb.set_trace()

      
        heads, arc_tags = parse.decode_MST(energy.data.cpu().numpy(), length, leading_symbolic=leading_symbolic,
                                           labeled=True)
        heads = from_numpy(heads)
        arc_tags = from_numpy(arc_tags)

        # compute the average score for each tree
        batch_size, max_len = heads.size()
        scores = torch.zeros_like(heads, dtype=energy.dtype, device=energy.device)
        for b_idx in range(batch_size):
            for len_idx in range(max_len):
                scores[b_idx, len_idx] = energy[b_idx, arc_tags[b_idx, len_idx], heads[b_idx, len_idx], len_idx]
        if mask is not None:
            scores = scores.sum(1) / mask.sum(1)
        else:
            scores = scores.sum(1) / max_len
        return heads, arc_tags, scores

    def decode_greedy(self, out_arc, out_arc_tag, mask, leading_symbolic):
        # '''
        # Args:
        #     out_arc: Tensor
        #         the arc scores with shape [batch_size, length, length]
        #     out_arc_tag: Tensor
        #         the labeled arc scores with shape [batch_size, length, arc_tag_space]
        #     mask: Tensor or None
        #         the mask tensor with shape = [batch_size, length]
        #     length: Tensor or None
        #         the length tensor with shape = [batch_size]
        #     leading_symbolic: int
        #         number of symbolic labels leading in arc_tag alphabets (set it to 0 if you are not sure)

        # Returns: (Tensor, Tensor)
        #         predicted heads and arc_tags.
        # '''
        def _decode_arc_tags(out_arc_tag, heads, leading_symbolic):
            # out_arc_tag shape [batch_size, length, arc_tag_space]
            arc_tag_h, arc_tag_c = out_arc_tag
            batch_size, max_len, _ = arc_tag_h.size()
            # create batch index [batch_size]
            batch_index = torch.arange(0, batch_size).type_as(arc_tag_h.data).long()
            # get vector for heads [batch_size, length, arc_tag_space],
            arc_tag_h = arc_tag_h[batch_index, heads.t()].transpose(0, 1).contiguous()
            # compute output for arc_tag [batch_size, length, num_arcs]
            out_arc_tag = self.bilinear(arc_tag_h, arc_tag_c)
            # remove the first #leading_symbolic arc_tags.
            out_arc_tag = out_arc_tag[:, :, leading_symbolic:]
            # compute the prediction of arc_tags [batch_size, length]
            _, arc_tags = out_arc_tag.max(dim=2)
            return arc_tags + leading_symbolic

        # out_arc shape [batch_size, length, length]
        out_arc = out_arc.data
        _, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        out_arc = out_arc + torch.diag(out_arc.new(max_len).fill_(-np.inf))
        # set invalid positions to -inf
        if mask is not None:
            # minus_mask = (1 - mask.data).byte().view(batch_size, max_len, 1)
            minus_mask = (1 - mask.data).byte().unsqueeze(2)
            out_arc.masked_fill_(minus_mask, -np.inf)

        # compute naive predictions.
        # predition shape = [batch_size, length]
        scores, heads = out_arc.max(dim=1)

        arc_tags = _decode_arc_tags(out_arc_tag, heads, leading_symbolic)

        # compute the average score for each tree
        if mask is not None:
            scores = scores.sum(1) / mask.sum(1)
        else:
            scores = scores.sum(1) / max_len
        return heads, arc_tags, scores

    def pre_loss(self, out_arc, out_arc_tag, heads=None, arc_tags=None, mask=None, length=None, use_log=True, temperature=1.0):  
        if (heads is not None and arc_tags is None) or (heads is None and arc_tags is not None):
            raise ValueError('heads and arc_tags should be both Nones or both not Nones')
        decode = True if (heads is None and arc_tags is None) else False
        softmax_func = F.log_softmax if use_log else F.softmax
        # out_arc shape [batch_size, length, length]
        # out_arc_tag shape [batch_size, length, arc_tag_space]
        ## Out_arc_tag [16,9,128]
        arc_tag_h, arc_tag_c = out_arc_tag
        # pdb.set_trace()
        # pdb.set_trace()
        batch_size, max_len, arc_tag_space = arc_tag_h.size()
        batch_index = None
        if not decode:
            if length is not None and heads.size(1) != max_len:
                heads = heads[:, :max_len]
                arc_tags = arc_tags[:, :max_len]
            # create batch index [batch_size]
            batch_index = torch.arange(0, batch_size).type_as(out_arc.data).long()
            # get vector for heads [batch_size, length, arc_tag_space],
            arc_tag_h = arc_tag_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()
        else:
            arc_tag_h = arc_tag_h.unsqueeze(2).expand(batch_size, max_len, max_len, arc_tag_space).contiguous()
            arc_tag_c = arc_tag_c.unsqueeze(1).expand(batch_size, max_len, max_len, arc_tag_space).contiguous()

        # compute output for arc_tag [batch_size, length, num_arcs]
        ## out_arc_tag.size() [16,9,27]
        ## We used gold head to calculate these scores
        out_arc_tag = self.bilinear(arc_tag_h, arc_tag_c)

        # mask invalid position to -inf for softmax_func
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            ## Out_arc = [16,9,9]
            ## Removed the right and bottom invalid columns and rows from out_arc
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if not decode:
            # loss_arc shape [batch_size, length, length]
            ## Please note that this is log softmax function
            out_arc = softmax_func(out_arc / temperature, dim=1)
            # loss_arc_tag shape [batch_size, length, num_arcs]
            out_arc_tag = softmax_func(out_arc_tag / temperature, dim=2)
            # mask invalid position to 0 for sum loss
            if mask is not None:
                out_arc = out_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
                out_arc_tag = out_arc_tag * mask.unsqueeze(2)

            # first create index matrix [length, batch_size]
            child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch_size)
            child_index = child_index.type_as(out_arc.data).long()
            # loss_arc shape [batch_size, length-1]
            ## This is the position we need to integrate constraints during training.

            out_arc = out_arc[batch_index, heads.data.t(), child_index][1:].t()
            # loss_arc_tag shape [batch_size, length-1]
            out_arc_tag = out_arc_tag[batch_index, child_index, arc_tags.data.t()][1:].t()
        else:
            # loss_arc shape [batch_size, length, length]
            out_arc = softmax_func(out_arc / temperature, dim=1)
            # loss_arc_tag shape [batch_size, length, length, num_arcs]
            out_arc_tag = softmax_func(out_arc_tag / temperature, dim=3).permute(0, 3, 1, 2)
        return out_arc, out_arc_tag

class BiRecurrentConv_Encoder(nn.Module):
    def __init__(self, word_dim, bert_dim, num_words, num_bert, char_dim, num_chars, use_bert, bert_outside_gpu, use_w2v, use_pos_type, use_case, use_number, use_gender, use_verb_form, use_full_pos, use_char, use_class, use_punc, pos_dim, num_pos, num_case, num_number, num_gender, num_verb, num_full, num_class, num_punc, num_filters,
                 kernel_size, rnn_mode, hidden_size, num_layers, embedd_word=None, embedd_bert=None, embedd_char=None, embedd_pos_type=None, embedd_case=None, embedd_gender=None, embedd_number=None, embedd_verb_form=None, embedd_full_pos=None, embedd_class=None, embedd_punc=None,
                 p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), initializer=None):
        super(BiRecurrentConv_Encoder, self).__init__()
        self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word) if use_w2v else None
        self.bert_embedd = Embedding(num_bert, bert_dim, init_embedding=embedd_bert) if (use_bert and not bert_outside_gpu) else None
        self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char) if use_char else None
        self.pos_type_embedd = Embedding(num_pos, pos_dim, init_embedding=embedd_pos_type) if use_pos_type else None
        self.case_embedd = Embedding(num_case, pos_dim, init_embedding=embedd_case) if use_case else None
        self.gender_embedd = Embedding(num_gender, pos_dim, init_embedding=embedd_gender) if use_gender else None
        self.number_embedd = Embedding(num_number, pos_dim, init_embedding=embedd_number) if use_number else None
        self.verb_form_embedd = Embedding(num_verb, pos_dim, init_embedding=embedd_verb_form) if use_verb_form else None
        self.full_pos_embedd = Embedding(num_full, pos_dim, init_embedding=embedd_full_pos) if use_full_pos else None
        self.class_embedd = Embedding(num_class, pos_dim, init_embedding=embedd_class) if use_class else None
        self.punc_embedd = Embedding(num_punc, pos_dim, init_embedding=embedd_punc) if use_punc else None        
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if use_char else None
        # self.OOV_layer = nn.Linear(word_dim, word_dim)
        # dropout word
        '''
        @note: this means that whole words are dropped, see https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html
        '''
        self.dropout_in = nn.Dropout2d(p_in)
        # standard dropout
        self.dropout_out = nn.Dropout2d(p_out)
        self.dropout_rnn_in = nn.Dropout(p_rnn[0])
        self.use_pos_type = use_pos_type
        self.use_case = use_case
        self.use_bert = use_bert
        self.bert_outside_gpu = bert_outside_gpu
        self.use_w2v = use_w2v
        self.use_number = use_number
        self.use_gender = use_gender
        self.use_verb_form = use_verb_form
        self.use_full_pos = use_full_pos        
        self.use_char = use_char
        self.use_class = use_class
        self.use_punc = use_punc
        self.rnn_mode = rnn_mode
        self.dim_enc = 0
        
        if use_w2v:
            self.dim_enc += word_dim
        if use_bert:
            self.dim_enc += bert_dim
        if use_pos_type:
            self.dim_enc += pos_dim 
        if use_case:
            self.dim_enc += pos_dim 
        if use_number:
            self.dim_enc += pos_dim 
        if use_gender:
            self.dim_enc += pos_dim 
        if use_verb_form:
            self.dim_enc += pos_dim 
        if use_full_pos:
            self.dim_enc += pos_dim 
        if use_class:
            self.dim_enc += pos_dim 
        if use_punc:
            self.dim_enc += pos_dim 

        if use_char:
            self.dim_enc += num_filters

        if rnn_mode == 'RNN':
            RNN = nn.RNN
            drop_p_rnn = p_rnn[1]
        elif rnn_mode == 'LSTM':
            RNN = nn.LSTM
            drop_p_rnn = p_rnn[1]
        elif rnn_mode == 'GRU':
            RNN = nn.GRU
            drop_p_rnn = p_rnn[1]
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)
        self.rnn = RNN(self.dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                       dropout=drop_p_rnn)
        self.initializer = initializer
        self.reset_parameters()

    def reset_parameters(self):
        if self.initializer is None:
            return

        for name, parameter in self.named_parameters():
            if name.find('embedd') == -1:
                if parameter.dim() == 1:
                    parameter.data.zero_()
                else:
                    self.initializer(parameter.data)

    def forward(self, input_word, input_bert, bert_vectors, input_char, input_pos_type, input_case, input_number, input_gender, input_verb_form, input_full_pos, input_class, input_punc, mask=None, length=None, hx=None):
        # hack length from mask
        # we do not hack mask from length for special reasons.
        # Thus, always provide mask if it is necessary.
        input = []
        if length is None and mask is not None:
            length = mask.data.sum(dim=1).long()
        if self.use_w2v:
            # [batch_size, length, word_dim]
            word = self.word_embedd(input_word)
            # apply dropout on input
            word = self.dropout_in(word)
            ###########################################
            ## To handle very less overlap of training and testing dataset
            # word = self.OOV_layer(word)
            input = word
        elif self.use_bert:
            if self.bert_outside_gpu:
                # [batch_size, length, pos_dim]
                word = bert_vectors
            else:
                word = self.bert_embedd(input_bert)
            word = self.dropout_in(word)
            input = word
                
        if self.use_char:
            # [batch_size, length, char_length, char_dim]
            char = self.char_embedd(input_char)
            char_size = char.size()
            # first transform to [batch *length, char_length, char_dim]
            # then transpose to [batch * length, char_dim, char_length]
            char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
            # put into cnn [batch*length, char_filters, char_length]
            # then put into maxpooling [batch * length, char_filters]
            char, _ = self.conv1d(char).max(dim=2)
            # reshape to [batch_size, length, char_filters]
            char = torch.tanh(char).view(char_size[0], char_size[1], -1)
            # apply dropout on input
            char = self.dropout_in(char)
            # when we don't have either bert or w2v, we need to create a new tensor
            if not self.use_bert and not self.use_w2v:
                input = char
            # Otherwise concatenate word and char [batch_size, length, word_dim+char_filter]
            else:
                input = torch.cat([input, char], dim=2)
            
        if self.use_bert and self.use_w2v:
            if self.bert_outside_gpu:
                word = bert_vectors
            else:
                word = self.bert_embedd(input_bert)
            # apply dropout on input
            word = self.dropout_in(word)
            input = torch.cat([input, word], dim=2)

        # TODO 10/6: Schauen wie das funktionieren kann:
        # if self.use_meanings:
        #     #[batch_size, length, meaning_length, meaning_dim]
        #     meaning = self.meaning_embedd(input_meaning)
        #     ms = torch.sum(meaning,axis=2) # > batch, length, meaning_dim; dieser Schritt nur dann wenn wir dynamisch die einzelnen Bedeutungen trainieren, nicht statisch! 
        #     ms = self.dropout_in(ms)
        #     input = torch.cat([input, ms], dim=2)
                
        if self.use_pos_type:
            # [batch_size, length, pos_dim]
            pos_type = self.pos_type_embedd(input_pos_type)
            # apply dropout on input
            pos_type = self.dropout_in(pos_type)
            input = torch.cat([input, pos_type], dim=2)
        if self.use_case:
            # [batch_size, length, pos_dim]
            case = self.case_embedd(input_case)
            # apply dropout on input
            case = self.dropout_in(case)
            input = torch.cat([input, case], dim=2)
        if self.use_gender:
            # [batch_size, length, pos_dim]
            gender = self.gender_embedd(input_gender)
            # apply dropout on input
            gender = self.dropout_in(gender)
            input = torch.cat([input, gender], dim=2)
        if self.use_number:
            # [batch_size, length, pos_dim]
            number = self.number_embedd(input_number)
            # apply dropout on input
            number = self.dropout_in(number)
            input = torch.cat([input, number], dim=2)
        if self.use_verb_form:
            # [batch_size, length, pos_dim]
            verb_form = self.verb_form_embedd(input_verb_form)
            # apply dropout on input
            verb_form = self.dropout_in(verb_form)
            input = torch.cat([input, verb_form], dim=2)
        if self.use_full_pos:
            # [batch_size, length, pos_dim]
            full_pos = self.full_pos_embedd(input_full_pos)
            # apply dropout on input
            full_pos = self.dropout_in(full_pos)
            input = torch.cat([input, full_pos], dim=2)
        if self.use_class:
            # [batch_size, length, pos_dim]
            clid = self.class_embedd(input_class)
            # apply dropout on input
            clid = self.dropout_in(clid)
            input = torch.cat([input, clid], dim=2)            
        if self.use_punc:
            # [batch_size, length, pos_dim]
            punc = self.punc_embedd(input_punc)
            # apply dropout on input
            punc = self.dropout_in(punc)
            input = torch.cat([input, punc], dim=2)
        # apply dropout rnn input
        # '''
        # @bug: oh? Warum wird hier noch mal ge-dropouted? alle embeddings hatten schon ihr individuelles dropout in den Zeilen darüber. Steht das im Paper?
        # ... ah, ich verstehe: self.dropout_in löscht ganze Wörter, dieser dropout aber nur einzelne Zellen
        # '''
        input = self.dropout_rnn_in(input) # schauen wie gross der tensor input hier ist
        # prepare packed_sequence
        if length is not None:
            seq_input, hx, rev_order, mask = utils.prepare_rnn_seq(input, length, hx=hx, masks=mask, batch_first=True) # b,s,e?
            self.rnn.flatten_parameters()
            seq_output, hn = self.rnn(seq_input, hx=hx)
            output, hn = utils.recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=True)
        else:
            # output from rnn [batch_size, length, hidden_size]
            self.rnn.flatten_parameters()
            output, hn = self.rnn(input, hx=hx)
        # apply dropout for the output of rnn
        output = self.dropout_out(output)
        # shortcut connection. Uncomment following 5 lines to disable.
        # input = input.permute(0,2,1)
        # input = torch.nn.functional.interpolate(input,size=output.shape[1]) # andere moeglichkeit: input lassen wie er ist, dafuer output[:,s-1]
        # input = input.permute(0,2,1)
        # input = torch.nn.functional.interpolate(input,size=output.shape[2])
        # output = torch.cat([output,input],dim=2) # output[:,:(s-1),:] nehmen oder alternativ output[:,1:,:]; RNN von pytorch koennte man auch nochmal mit shortcuts zwischen den layern versuchen. 
        return output, hn, mask, length

class Gating(nn.Module):
    # Implementation of:
    # Sato, Motoki, et al. "Adversarial training for cross-domain universal dependency parsing."
    # Proceedings of the CoNLL 2017 Shared Task: Multilingual Parsing from Raw Text to Universal
    #  Dependencies. 2017.
    def __init__(self, num_gates, input_dim):
        super(Gating, self).__init__()
        self.num_gates = num_gates
        self.input_dim = input_dim
        if self.num_gates == 2:
            self.linear = nn.Linear(self.num_gates * self.input_dim, self.input_dim)
        elif self.num_gates > 2:
            self.linear = nn.Linear(self.num_gates * self.input_dim, self.num_gates * self.input_dim)
            self.softmax = nn.Softmax(-1)
        else:
            raise ValueError('num_gates should be greater or equal to 2')

    def forward(self, tuple_of_inputs):
        # output size should be equal to the input sizes
        if self.num_gates == 2:
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            output = torch.mul(alpha, tuple_of_inputs[0]) + torch.mul(1 - alpha, tuple_of_inputs[1])
        else: # elif self.num_gates > 2:
            # extend the gating mechanism to more than 2 encoders
            batch_size, len_size, dim_size = tuple_of_inputs[0].size()
            # '''
            # @bug: oh: normalerweise ist der input für softmax ein linear layer, d.h. keine Aktivierungsfunktion wie sigmoid, sondern nur W*a + bias
            #      => sigmoid weg?
            # '''
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            alpha = self.softmax(alpha.view(batch_size, len_size, dim_size, self.num_gates))
            output = torch.sum(torch.mul(alpha,torch.stack(tuple_of_inputs, dim=-1)), dim=-1)
        return output
