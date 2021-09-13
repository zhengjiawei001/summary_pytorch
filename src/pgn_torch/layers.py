# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 8:31 下午
# @Author  : zhengjiawei
# @FileName: layers.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pgn_torch.utils import replace_oovs
from src.utils.params_utils import get_params
from src.utils.wv_loader import load_embedding_matrix


class Encoder(nn.Module):
    def __init__(self, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=True)

        self.lstm = nn.LSTM(self.embedding_dim, self.enc_units, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)

        enc_output, hidden_state = self.lstm(x)

        return enc_output, hidden_state

    def initialize_hidden_state(self):
        return torch.zeros((2, self.batch_size, self.enc_units))


class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.w_s = nn.Linear(2 * units, 2 * units)
        self.w_h = nn.Linear(2 * units, 2 * units)
        self.w_c = nn.Linear(1, 2 * units)
        self.v = nn.Linear(2 * units, 1)

    def forward(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=False, prev_coverage=None):
        """
        Args:
        decoder_states (tuple): each with shape (2, batch_size, hidden_units)
        encoder_output (Tensor): shape (batch_size, seq_len, hidden_units).
        x_padding_masks (Tensor): shape (batch_size, seq_len).
        coverage_vector (Tensor): shape (batch_size, seq_len).
        Returns:
        context_vector (Tensor): shape (batch_size, 2*hidden_units).
        attention_weights (Tensor): shape (batch_size, seq_length).
        coverage_vector (Tenso): shape (batch_size, seq_length).

        """
        # (2,batch_size,hidden_units) -> (batch_size,1,2*hidden_units)
        # h_dec, c_dec = dec_hidden

        dec_hidden = dec_hidden.transpose(0, 1).contiguous().reshape(dec_hidden.size(1), 1, -1)

        # 计算attention
        enc_featuers = self.w_h(enc_output.contiguous())  # wh*(batch_size,seq_length,2*hidden_units)
        dec_features = self.w_s(dec_hidden)  # Ws s_t (batch_size, seq_length, 2*hidden_units)
        # print('enc_featuers.size:', enc_featuers.size())
        # print('dec_features.size:', dec_features.size())
        att_inputs = enc_featuers + dec_features  # # (batch_size, seq_length, 2*hidden_units)
        # 增加coverage 向量
        if use_coverage and prev_coverage is not None:
            coverage_features = self.w_c(
                prev_coverage.unsqueeze(
                    2))  # (batch_size, seq_len,1) * (1, 2 * units) ->(batch_size, seq_len,2 * units)
            att_inputs = att_inputs + coverage_features

        # 求attention 概率分布
        score = self.v(torch.tanh(att_inputs))  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(score, dim=1).squeeze(-1)  # # (batch_size, seq_length)
        # print('attention_weights:', attention_weights.size())
        # print('enc_pad_mask:', enc_pad_mask.size())
        attention_weights = attention_weights * enc_pad_mask

        # Normalize attention weights after excluding padded positions.
        normalization_factor = attention_weights.sum(1, keepdim=True)

        attention_weights = attention_weights / normalization_factor

        context_vector = torch.bmm(attention_weights.unsqueeze(1), enc_output)  # (batch_size, 1, 2*hidden_units)
        context_vector = context_vector.squeeze(1)  # (batch_size, 2*hidden_units)

        # Update coverage vector.
        if use_coverage:
            coverage_vector = prev_coverage + attention_weights
        else:
            coverage_vector = []
        return context_vector, attention_weights, coverage_vector


class Decoder(nn.Module):
    def __init__(self, embedding_matrix, batch_size, dec_units, is_cuda=True, pointer=True):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.pointer = pointer
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=True)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        # self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.embedding_dim, self.dec_units, batch_first=True, bidirectional=True)

        self.W1 = nn.Linear(self.dec_units * 4, self.dec_units)
        self.W2 = nn.Linear(self.dec_units, self.vocab_size)
        if self.pointer:
            self.w_gen = nn.Linear(self.dec_units * 4 + self.embedding_dim, 1)

    def forward(self, x, dec_states, context_vector):
        """
         Args:
             x (Tensor): shape (batch_size, 1).
             dec_states :each with shape (2, batch_size, hidden_units) for each.
             context_vector (Tensor): shape (batch_size,2*hidden_units).
         Returns:
             p_vocab (Tensor): shape (batch_size, vocab_size).
             decoder_states (tuple): The lstm states in the decoder.Each with shapes (2, batch_size, hidden_units).
             p_gen (Tensor): shape (batch_size, 1).

         """
        x = self.embedding(x)  # (batch_size, 1) -> (batch,embedding_dim)

        decoder_output, decoder_states = self.lstm(x, dec_states)

        # 拼接 状态向量 和 上下文向量
        # (batch_size,1,2*hidden_size) ->(batch_size,2*hidden_size)
        decoder_output = decoder_output.view(-1, 2 * self.dec_units)
        # batch_size,2*hidden_size+batch_size,2*hidden_units =(batch_size,4*hidden_units)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)

        FF1_out = self.W1(concat_vector)  # (batch_size, hidden_units)
        FF2_out = self.W2(FF1_out)  # (batch_size, vocab_size)

        p_vocab = F.softmax(FF2_out, dim=1)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        h_dec = h_dec.transpose(0, 1).reshape(h_dec.size(1), 1, -1)
        # s_t = torch.cat([h_dec, c_dec], dim=2)  # (1, batch_size, 2*hidden_units)

        p_gen = None
        if self.pointer:
            # print('size:', context_vector.size(), h_dec.squeeze(1).size(), x.squeeze(1).size())
            x_gen = torch.cat([context_vector, h_dec.squeeze(1), x.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))
        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)


class PGN(nn.Module):
    def __init__(self, vocab):
        super(PGN, self).__init__()
        params = get_params()
        self.vocabs = vocab
        self.embedding_matrix = load_embedding_matrix()
        self.vocab_size, self.embedding_dim = self.embedding_matrix.shape
        self.device = params['device']
        self.batch_size = params['batch_size']
        self.enc_units = params['enc_units']
        self.dec_units = params['dec_units']
        self.pointer = params['pointer_gen']
        self.eps = 0.001
        self.coverage = params['use_coverage']
        self.attention = BahdanauAttention(params['attn_units'])

        self.encoder = Encoder(self.embedding_matrix, self.enc_units, self.batch_size)

        self.decoder = Decoder(self.embedding_matrix, self.batch_size, self.dec_units)

        self.reduce_state = ReduceState()

    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        """
        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.
        Returns:
            final_distribution (Tensor): shape (batch_size, )
        """
        if not self.pointer:
            return p_vocab

        batch_size = x.size(0)

        # Clip the probabilities.
        p_gen = torch.clamp(p_gen, 0.001, 0.999)

        p_vocab_weighted = p_gen * p_vocab  # Get the weighted probabilities.

        attention_weighted = (1 - p_gen) * attention_weights  # (batch_size, seq_len)

        # 得到 词典 和 oov 的总体概率分布
        # print('batch_size:',batch_size)
        #
        # max_oov = max_oov.item()
        # print('max_oov',max_oov)
        extension = torch.zeros((batch_size, max_oov)).to(self.device)

        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)  # (batch_size, extended_vocab_size)

        # 将attention_weighted中的数据，按照x中的索引位置，加到p_vocab_extended中
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)

        return final_distribution

    def forward(self, x, y, len_oovs, teacher_forcing):
        """
        Args:
            x (Tensor): shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor): shape (bacth_size, y_len)
            len_oovs (Tensor)
            batch (int)
            num_batches(int)
            teacher_forcing(bool)
        """

        x_copy = replace_oovs(x, self.vocabs)

        # torch.ne 计算是否不等于，x中的元素不等于0 返回True,等于返回False
        x_padding_masks = torch.ne(x, 0).float()
        encoder_output, encoder_states = self.encoder(x_copy)

        decoder_states = encoder_states
        print('encoder_states:',encoder_states[0].size())
        coverage_vector = torch.zeros_like(x).to(self.device).float()
        # print('coverage_vector:',coverage_vector.size())
        step_losses = []
        x_t = y[:, 0]
        for t in range(y.shape[1] - 1):
            if teacher_forcing:
                x_t = y[:, t]
            x_t = replace_oovs(x_t, self.vocabs)
            y_t = y[:, t + 1]

            context_vector, attention_weights, coverage_vector = self.attention(decoder_states[0],
                                                                                encoder_output,
                                                                                x_padding_masks,
                                                                                use_coverage=True,
                                                                                prev_coverage=coverage_vector)

            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1), decoder_states, context_vector)

            final_dist = self.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))

            x_t = torch.argmax(final_dist, dim=1).to(self.device)

            if not self.pointer:
                y_t = replace_oovs(y_t, self.vocabs)
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)
            mask = torch.ne(y_t, 0).float()
            loss = -torch.log(target_probs + self.eps)
            if self.coverage:
                ct_min = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss + cov_loss  # loss = loss * self.LAMBDA * cov_loss
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)

        seq_len_mask = torch.ne(y, 0).float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss
