# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 10:25 上午
# @Author  : zhengjiawei
# @FileName: seq2seq_batcher.py
# @Software: PyCharm

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.build_data import load_dataset
from src.utils import config


def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=50, buffer_size=5, sample_sum=None, ):
    # 如果只是训练预测 则用两部分数据集，如果想要验证模型效果，则用三部分数据集
    # 如果之前已经将数据切分了，那么只需要按照不同路径读入就好，
    train_x, train_y = load_dataset(config.train_x_path, config.train_y_path,
                                    max_enc_len, max_dec_len)
    val_x, val_y = load_dataset(config.test_x_path, config.test_y_path,
                                max_enc_len, max_dec_len)
    # train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=0.8)
    if sample_sum:
        train_x = train_x[:sample_sum]
        train_y = train_y[:sample_sum]
    # print(type(train_x))
    # print(train_x)
    print(f'total {len(train_y)} examples ...')
    # print('torch.from_numpy(train_x):', torch.from_numpy(train_x)[0])
    # print('torch.from_numpy(train_y):', torch.from_numpy(train_y)[0])
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    # val_dataset = TensorDataset(val_x,val_y)
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_steps_per_epoch = len(train_x) // batch_size
    val_steps_per_epoch = len(val_x) // batch_size
    return train_loader, val_loader, train_steps_per_epoch, val_steps_per_epoch


def beam_test_batch_generator(beam_size, max_enc_len=200, max_dec_len=50):
    # 加载数据
    test_x, _ = load_dataset(config.test_x_path, config.test_y_path,
                             max_enc_len, max_dec_len)
    print(f'total {len(test_x)} test examples ...')
    for row in tqdm(test_x, total=len(test_x), desc='Beam Search'):
        beam_search_data = torch.tensor([row for i in range(beam_size)])
        yield beam_search_data


if __name__ == '__main__':
    beam_test_batch_generator(4)
