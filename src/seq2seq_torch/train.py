# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 3:05 下午
# @Author  : zhengjiawei
# @FileName: train.py
# @Software: PyCharm
from src.seq2seq_torch.seq2seq_model import Seq2Seq
from src.seq2seq_torch.train_helper import train_model
from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab


def train(params):
    config_gpu()
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count
    # 构建模型
    print("Building the model ...")
    model = Seq2Seq(params, vocab)
    model.to(params['device'])
    train_model(model, vocab, params)


if __name__ == '__main__':
    # load params
    params = get_params()

    params['mode'] = 'train'

    # train
    train(params)
