# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 3:13 下午
# @Author  : zhengjiawei
# @FileName: train_helper.py
# @Software: PyCharm
import time
from functools import partial

import torch

from src.seq2seq_torch.seq2seq_batcher import train_batch_generator


def train_model(model, vocab, params):
    epochs = params['epochs']
    pad_index = vocab.word2id[vocab.PAD_TOKEN]

    # 获取vocab大小
    params['vocab_size'] = vocab.count

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_load, val_dataset, train_steps_per_epoch, val_steps_per_epoch = train_batch_generator(
        params['batch_size'], params['max_enc_len'], params['max_dec_len'])

    for epoch in range(epochs):
        start = time.time()
        enc_hidden = model.encoder.initialize_hidden_state().to(params['device'])
        # print('enc_hidden:', enc_hidden)
        total_loss = 0.
        running_loss = 0.
        for (batch, (inputs, target)) in enumerate(train_load):
            inputs = inputs.to(params['device'])
            target = target.to(params['device'])
            # print('inputs:', inputs)
            # print('target:', target)
            batch_loss = train_step(model, inputs, target,
                                    enc_hidden,
                                    loss_function=partial(loss_function, pad_index=pad_index),
                                    optimizer=optimizer, mode='train')
            total_loss += batch_loss.cpu().item()

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.8f}'.format(epoch + 1,
                                                             batch,
                                                             (total_loss - running_loss) / 50))
                running_loss = total_loss

        if (epoch + 1) % 2 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': total_loss}, params['checkpoint_dir'])

            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                params['checkpoint_dir']))

        valid_loss = evaluate(model, val_dataset, val_steps_per_epoch,
                              loss_func=partial(loss_function, pad_index=pad_index))

        print('Epoch {} Loss {:.4f}; val Loss {:.4f}'.format(
            epoch + 1, total_loss / train_steps_per_epoch, valid_loss)
        )

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def train_step(model, enc_input, dec_target, enc_hidden, loss_function=None, optimizer=None, mode='train'):
    if mode == 'train':
        model.train()


    # 第一个隐藏层输入
    # dec_hidden = enc_hidden
    # 逐个预测序列
    # enc_input, enc_hidden, dec_hidden, dec_target

    pred, _ = model(enc_input, enc_hidden, dec_target)
    # pred = pred.permute(0, 2, 1)
    # print('pred:', pred.shape)
    # print('dec_target[:, 1:]', dec_target[:, 1:].shape)
    batch_loss = loss_function(dec_target[:, 1:], pred)
    if mode == 'train':
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return batch_loss


def loss_function(real, pred, pad_index):
    loss_object = torch.nn.CrossEntropyLoss(ignore_index=pad_index)
    vocab_size = pred.size(2)
    real = real.reshape(-1)
    pred = pred.reshape(-1, vocab_size)
    return loss_object(pred, real)


def evaluate(model, val_dataset, val_steps_per_epoch, loss_func):
    print('Starting evaluate ...')
    model.eval()
    total_loss = 0.
    enc_hidden = model.encoder.initialize_hidden_state().cuda()
    with torch.no_grad():
        for (batch, (inputs, target)) in enumerate(val_dataset, start=1):
            inputs = inputs.cuda()
            target = target.cuda()
            batch_loss = train_step(model, inputs, target, enc_hidden,
                                    loss_function=loss_func, mode='val')
            total_loss += batch_loss
    return total_loss / val_steps_per_epoch
