# -*- coding: utf-8 -*-
import math
from builtins import object
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

from src.pgn_torch.utils import sort_batch_by_len
from src.utils.config import vocab_path
from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab


def article_to_ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # if w is oov
            if w not in oovs:  # add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract_to_ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word_to_id(Vocab.UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word_to_id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def get_steps_per_epoch(params):
    if params['mode'] == 'train':
        file = open(params['train_seg_y_dir'])
    elif params['mode'] == 'test':
        file = open(params['test_seq_x_dir'])
    else:
        file = open(params['val_seg_x_dir'])
    num_examples = len(file.readlines())
    if params['decode_mode'] == 'beam':
        return num_examples
    steps_per_epoch = math.ceil(num_examples // params['batch_size'])
    return steps_per_epoch


def get_enc_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
    sequence: List of ids (integers)
    max_len: integer
    start_id: integer
    stop_id: integer
    Returns:
    inp: sequence length <=max_len starting with start_id
    target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    if len(inp) > max_len:
        inp = inp[:max_len]
    else:
        inp.append(stop_id)
    return inp


class PairDataset(object):
    def __init__(self,
                 params,
                 mode,
                 vocab,
                 max_enc_len,
                 max_dec_len,
                 batch_size,
                 ):
        print("Reading dataset...")
        self.output = defaultdict(list)
        if mode != 'test':
            file_x = params[f"{mode}_seg_x_dir"]
            file_y = params[f"{mode}_seg_y_dir"]
            file_x = open(file_x, encoding='utf-8')
            file_y = open(file_y, encoding='utf-8')
            x_data = file_x.readlines()
            y_data = file_y.readlines()
            for raw_record in zip(x_data, y_data):
                article = raw_record[0]

                start_decoding = vocab.word_to_id(Vocab.START_DECODING)
                stop_decoding = vocab.word_to_id(Vocab.STOP_DECODING)

                article_words = article.split()[:max_enc_len]

                enc_input = [vocab.word_to_id(w) for w in article_words]
                enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

                # add start and stop flag
                enc_input = get_enc_inp_targ_seqs(enc_input, max_enc_len, start_decoding, stop_decoding)
                enc_input_extend_vocab = get_enc_inp_targ_seqs(enc_input_extend_vocab,
                                                               max_enc_len, start_decoding,
                                                               stop_decoding)

                # mark长度
                enc_len = len(enc_input)
                # 添加mask标记
                encoder_pad_mask = [1 for _ in range(enc_len)]

                abstract = raw_record[1]
                abstract_words = abstract.split()
                abs_ids = [vocab.word_to_id(w) for w in abstract_words]

                dec_input, target = get_dec_inp_targ_seqs(abs_ids, max_dec_len, start_decoding, stop_decoding)

                if params['pointer_gen']:
                    abs_ids_extend_vocab = abstract_to_ids(abstract_words, vocab, article_oovs)
                    _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab, max_dec_len, start_decoding, stop_decoding)

                # mark 长度
                dec_len = len(target)
                # 添加mark 标记
                decoder_pad_mask = [1 for _ in range(dec_len)]

                self.output['enc_len'].append(enc_len)
                self.output['enc_input'].append(enc_input)

                self.output['enc_input_extend_vocab'].append(enc_input_extend_vocab)
                self.output['article_oovs'].append(article_oovs)
                self.output['oovs_length'].append(len(article_oovs))

                self.output['dec_input'].append(dec_input)
                self.output['target'].append(target)
                self.output['dec_len'].append(dec_len)
                self.output['article'].append(article)

                self.output['abstract'].append(abstract)
                self.output['abstract_sents'].append(abstract)
                self.output['decoder_pad_mask'].append(decoder_pad_mask)
                self.output['encoder_pad_mask'].append(encoder_pad_mask)
        else:
            file_x = params[f"{mode}_seg_x_dir"]
            file_x = open(file_x, encoding='utf-8')
            x_data = file_x.readlines()
            for raw_record in x_data:
                article = raw_record

                start_decoding = vocab.word_to_id(Vocab.START_DECODING)
                stop_decoding = vocab.word_to_id(Vocab.STOP_DECODING)

                article_words = article.split()[:max_enc_len]

                enc_input = [vocab.word_to_id(w) for w in article_words]
                enc_input_extend_vocab, article_oovs = article_to_ids(article_words, vocab)

                # add start and stop flag
                enc_input = get_enc_inp_targ_seqs(enc_input, max_enc_len, start_decoding, stop_decoding)
                enc_input_extend_vocab = get_enc_inp_targ_seqs(enc_input_extend_vocab,
                                                               max_enc_len, start_decoding,
                                                               stop_decoding)

                # mark长度
                enc_len = len(enc_input)
                # 添加mask标记
                encoder_pad_mask = [1 for _ in range(enc_len)]

                # mark 长度
                # 添加mark 标记

                self.output['enc_len'].append(enc_len)
                self.output['enc_input'].append(enc_input)

                self.output['enc_input_extend_vocab'].append(enc_input_extend_vocab)
                self.output['article_oovs'].append(article_oovs)
                self.output['oovs_length'].append(len(article_oovs))
                self.output['dec_input'].append([0])
                self.output['encoder_pad_mask'].append(encoder_pad_mask)


class SampleDataset(Dataset):
    def __init__(self, params, mode, vocab, PairDataset, is_training: bool = True):
        pair = PairDataset(params, mode=mode, vocab=vocab, max_enc_len=params['max_enc_len'],
                           max_dec_len=params['max_dec_len'], batch_size=params['batch_size'])
        self.is_training = is_training
        self.output = pair.output
        self.mode = mode

    def __getitem__(self, index):

        return {
            'x': self.output['enc_input'][index],
            'OOV': self.output['article_oovs'][index],
            'len_OOV': self.output['oovs_length'][index],
            'y': self.output['dec_input'][index],
            'x_len': len(self.output['enc_input'][index]),
            'y_len': len(self.output['dec_input'][index]),
        }

    def __len__(self):
        return len(self.output['enc_input'])


def collate_fn(batch):
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])

    return x_padded, y_padded, x_len, y_len, OOV, len_OOV # 除了 OOV 是个列表外， 其它都是tensor


def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    """
    Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).
    Args:
      sequence: List of ids (integers)
      max_len: integer
      start_id: integer
      stop_id: integer
    Returns:
      inp: sequence length <=max_len starting with start_id
      target: sequence same length as input, ending with stop_id only if there was no truncation
    """
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:
        inp = inp[:max_len]
        target = target[:max_len]
    elif len(inp) == max_len:
        target.append(stop_id)
    else:
        target.append(stop_id)
        inp.append(stop_id)
    return inp, target


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 获取参数
    params = get_params()
    params['mode'] = 'test'
    vocab = Vocab(vocab_path)
    train_data = SampleDataset(params, params['mode'], vocab, PairDataset)
    train_iter = DataLoader(dataset=train_data,
                            batch_size=3,
                            shuffle=True,
                            collate_fn=collate_fn)
    ds = iter(train_iter)
    print(next(ds))
