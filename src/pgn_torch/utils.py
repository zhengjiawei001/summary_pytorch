# -*- coding: utf-8 -*-
import heapq
import random
import time
from builtins import object

import numpy as np
import torch

from ..utils.config import vocab_path
from ..utils.params_utils import get_params


def replace_oovs(in_tensor, vocab):
    params = get_params()
    # 用 vocab.UNK 填充in_tensor.shape 的矩阵

    full_num = vocab.word2id[vocab.UNKNOWN_TOKEN]
    oov_token = torch.full(in_tensor.size(), full_num).long().to(params['device'])

    # torch.where()函数的作用是按照一定的规则合并两个tensor类型。in_tensor> len(vocab) - 1的保留
    len_voceb = len(vocab.load_vocab(vocab_path)[0])
    out_tensor = torch.where(in_tensor > len_voceb - 1, oov_token, in_tensor)
    return out_tensor


def sort_batch_by_len(data_batch):
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': [],
           }
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # 根据 x_len 的长度来重排
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()
    data_batch = {name: [_tensor[i] for i in sorted_indices]
                  for name, _tensor in res.items()}
    return data_batch


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


class ScheduledSampler:
    def __init__(self, phases):
        self.phases = phases
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        sampling_prob = random.random()

        if sampling_prob >= self.scheduled_probs[phase]:
            return True
        else:
            return False


class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        The scores are calculated according to the definitions in
        https://opennmt.net/OpenNMT/translation/beam_search/.
        1. Lenth normalization is used to normalize the cumulative score
        of a whole sequence.
        2. Coverage normalization is used to favor the sequences that fully
        cover the information in the source. (In this case, it serves different
        purpose from the coverage mechanism defined in PGN.)
        3. Alpha and beta are hyperparameters that used to control the
        strengths of ln and cn.
        """
        len_Y = len(self.tokens)
        # Lenth normalization
        # Beam search

        ln = (5 + len_Y) ** 0.2 / (5 + 1) ** 0.2
        cn = 0.2 * torch.sum(  # Coverage normalization
            torch.log(
                1e-31 +
                torch.where(
                    self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(config.DEVICE))
                )
            )
        )

        score = sum(self.log_probs) / ln + cn
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


def add2heap(heap, item, k):
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)


# def result_index2text(hyp, vocab, batch):
#     article_oovs = batch[0]["article_oovs"].numpy()[0]
#     hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
#     hyp.article = batch[0]["article"].numpy()[0].decode()
#
#     words = []
#     for index in hyp.tokens:
#         if index != vocab.START_DECODING_INDEX and index != vocab.STOP_DECODING_INDEX:
#             if index < (len(article_oovs) + vocab.size()):
#                 if index < vocab.size():
#                     words.append(vocab.id_to_word(index))
#                 else:
#                     words.append(article_oovs[index - vocab.size()].decode())
#             else:
#                 print('error values id :{}'.format(index))
#     hyp.abstract = " ".join(words)
#     return hyp
