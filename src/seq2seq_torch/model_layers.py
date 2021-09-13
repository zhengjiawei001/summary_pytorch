# -*- coding: utf-8 -*-
# from src.utils.gpu_utils import config_gpu

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab, load_embedding_matrix


class Encoder(nn.Module):
    def __init__(self, embedding_matrix, dec_units, batch_sz):
        super().__init__()
        self.batch_size = batch_sz
        self.dec_units = dec_units
        _, self.embedding_dim = embedding_matrix.shape
        # 记载的矩阵需要有dim属性，所有转换成tensor
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=True)
        self.gru = nn.GRU(self.embedding_dim, self.dec_units, batch_first=True,
                          bidirectional=True)

    def forward(self, x, hidden):
        x = self.embedding(x)  # (batch_size,seq_len,embedding_dim)
        # x:(batch_size,seq_len,embedding_dim)

        outputs, hidden_state = self.gru(x, hidden)
        # outputs(batch_size,sequence length,dec_units)
        # hidden_state (2,batch_size, dec_units)

        hidden_state = hidden_state.transpose(0, 1).contiguous().reshape(hidden_state.size(1),
                                                                         -1)  # (batch_size, hidden_dim * 2)

        return outputs, hidden_state

    def initialize_hidden_state(self):
        return torch.zeros((2, self.batch_size, self.dec_units))


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.w2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query, values):
        # query为上次的GRU隐藏层
        # values是encoder部分的输出enc_output
        # 在seq2seq模型中，st是后面的query向量，而编码过程的隐藏状态hi是values。

        # hidden shape ==(batch_size,dec_units)   (batch_size, hidden_dim * 2)
        # hidden_with_time_axis shape == (batch_size, 1, 2*hidden_dim)
        # we are doing this to perform addition to calculate the score

        hidden_with_time_axis = torch.unsqueeze(query, 1)
        # score shape == (batch_size,max_length,1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        # 计算注意力权重值
        # values shape == ()
        score = self.v(torch.tanh(
            self.w1(values) + self.w2(hidden_with_time_axis)))
        # (batch_size, seq_len, hidden_dim)+(batch_size, 1,hidden_dim) 经过tanh之后
        # 纬度不发生变化，还是(batch_size, seq_len, hidden_dim)
        # attention_weights shape == (batch_size, max_length, 1)

        attention_weights = F.softmax(score, dim=1)

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, dec_units)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_size = batch_sz
        self.dec_units = dec_units
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=True)
        self.gru = nn.GRU(self.embedding_dim, self.dec_units, batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(self.dec_units * 2, self.vocab_size)  # self.vocab_size
        # self.change_dim = nn.Linear()
        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def forward(self, x, hidden, enc_output):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, seq_len, dec_units * 2)
        # hidden shape== (batch_size, hidden_dim * 2)
        context_vector, attention_weights = self.attention(hidden,
                                                           enc_output)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # 将上一循环的预测结果跟注意力权重结合在一起作为本次GRU网络输入
        # context_vector (batch_size, dec_units)
        # x shape for after concatenation ==(batch_size,1,embedding_dim+dec_units)

        x = torch.cat([torch.unsqueeze(context_vector, 1), x], dim=-1)
        # passing the concatenated vector to the GRU

        change_dim = nn.Linear(x.shape[-1], self.embedding_dim).cuda()
        x = change_dim(x)
        # (batch_size, hidden_dim * 2) -> (2,batch_size, hidden_dim)
        hidden = hidden.reshape(-1, 2, self.dec_units).permute(1, 0, 2).contiguous()
        output, hidden_state = self.gru(x, hidden)

        # output shape == (batch_size,vocab)
        output = output.reshape(-1, output.shape[2])
        prediction = self.fc(output)
        # prediction = F.softmax(prediction, dim=0)
        # print('hidden_state.shape...:',hidden_state.shape)
        return prediction, hidden_state, attention_weights


if __name__ == '__main__':
    # GPU资源配置
    # config_gpu()
    # 获得参数
    params = get_params()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 计算vocab size
    vocab_size = vocab.count
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()
    # print(embedding_matrix.shape)
    # print(embedding_matrix.size)
    input_sequence_len = 250
    batch_size = 64
    embedding_dim = 500
    hidden_dim = 1024

    # 编码器结构 embedding_matrix, enc_units, batch_sz
    sample_embed_matrix = torch.randn((256, 512))
    encoder = Encoder(embedding_matrix, hidden_dim, batch_size)
    # example_input
    example_input_batch = torch.ones((batch_size, input_sequence_len)).long()
    # sample input
    sample_hidden = encoder.initialize_hidden_state()

    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, hidden_dim) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, hidden_dim) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(hidden_dim=hidden_dim)
    print(sample_hidden.size())
    print(sample_output.size())
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    # sample_hidden = sample_hidden.cuda()
    # sample_output = sample_output.cuda()
    x = torch.randint(sample_embed_matrix.size(0), (batch_size, 1))  # .cuda()
    decoder = Decoder(embedding_matrix, hidden_dim, batch_size)
    sample_decoder_output, state, attention_weights = decoder(
        x,
        sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
