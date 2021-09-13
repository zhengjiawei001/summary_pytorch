# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pandas import np

from src.seq2seq_torch.model_layers import Encoder, BahdanauAttention, Decoder
from src.utils.params_utils import get_params
from src.utils.wv_loader import load_embedding_matrix, Vocab


class Seq2Seq(nn.Module):
    def __init__(self, params, vocab):
        super().__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.vocab = vocab
        self.batch_size = params['batch_size']
        self.enc_units = params['enc_units']
        self.dec_units = params['dec_units']
        self.attn_units = params['attn_units']
        self.encoder = Encoder(self.embedding_matrix,
                               self.enc_units,
                               self.batch_size)
        self.attention = BahdanauAttention(self.attn_units)

        self.decoder = Decoder(self.embedding_matrix,
                               self.dec_units,
                               self.batch_size)

    def forward(self, enc_input, enc_hidden, dec_target):
        enc_output, enc_hidden = self.encoder(enc_input, enc_hidden)
        return self.teacher_decoder(enc_hidden, enc_output, dec_target)

    def teacher_decoder(self, dec_hidden, enc_output, dec_target):
        predictions = []

        dec_input = torch.from_numpy(np.expand_dims([self.vocab.START_DECODING_INDEX] * self.batch_size, axis=1))
        dec_input = dec_input.cuda()
        #  Teacher forcing- feeding the target as the next input
        # dec_input = dec_input.to(self.params['device'])
        # print('dec_target:', dec_target)
        for t in range(1, dec_target.shape[1]):
            # passing enc_output to the decoder
            # print('1   dec_hidden.shape:', dec_hidden.shape)
            pred, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
            # print('2   dec_hidden.shape:', dec_hidden.shape)
            dec_hidden = dec_hidden.permute(1, 0, 2).reshape(dec_hidden.size(1), -1)

            dec_input = torch.unsqueeze(dec_target[:, t], 1)
            # print('pred:', pred.shape)
            predictions.append(pred)

        return torch.stack(predictions, 1), dec_hidden


if __name__ == '__main__':
    # GPU资源配置

    # 获得参数
    params = get_params()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 计算vocab size
    # vocab_size = vocab.count
    input_sequence_len = 200

    params = {"vocab_size": vocab.count,
              "embed_size": 500,
              "enc_units": 512,
              "attn_units": 512,
              "dec_units": 512,
              "batch_size": 128,
              "input_sequence_len": input_sequence_len}

    model = Seq2Seq(params, vocab).cuda()

    # example_input
    example_input_batch = torch.ones((params['batch_size'], params['input_sequence_len'])).long().cuda()

    # sample input
    print('exp_input_batch.size', example_input_batch.size())
    sample_hidden = model.encoder.initialize_hidden_state().cuda()
    print('sample_hidden.size', sample_hidden.size())

    sample_output, sample_hidden = model.encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(params['attn_units']).cuda()
    sample_output = sample_output.cuda()
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    x = torch.randint(0, params['vocab_size'], (params['batch_size'], 1)).long().cuda()

    sample_decoder_output, _, _ = model.decoder(
        x,
        sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
