# -*- coding : utf-8 -*-
# @Time      : 2020/4/4 15:31
# @Author    : Xu Huipeng
# @Github    : https://github.com/Brycexxx

import heapq
import logging
import os
import random
import shutil
from collections import Counter
from collections import namedtuple
from json import dumps
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ujson as json
from rouge import Rouge
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from tqdm import tqdm

"""
0-pad
1-unk
2-bos
3-eos
"""


class LCSTS(Dataset):

    def __init__(self, data_path: str, is_training: bool = True):
        self.is_training = is_training
        data = torch.load(data_path)
        self.article_idxs = data['article_idxs']
        self.article_extend_idxs = data['article_extend_idxs']
        self.summary_idxs = data['summary_idxs']
        self.summary_extend_idxs = data['summary_extend_idxs']
        if is_training:
            self.oovs_length = data['oovs_length']
        else:
            self.summaries = data['summaries']
            self.articles = data['articles']
            self.oovs_length = [len(oovs) for oovs in data['article_oovs']]
            self.article_oovs = data['article_oovs']

    def __getitem__(self, idx):
        if self.is_training:
            return (self.article_idxs[idx].long(), self.article_extend_idxs[idx].long(),
                    self.summary_idxs[idx].long(), self.summary_extend_idxs[idx].long(),
                    self.oovs_length[idx])
        else:
            return (self.article_idxs[idx].long(), self.article_extend_idxs[idx].long(),
                    self.summary_idxs[idx].long(), self.summary_extend_idxs[idx].long(),
                    self.oovs_length[idx], self.summaries[idx], self.articles[idx],
                    self.article_oovs[idx])

    def __len__(self):
        return self.article_idxs.size(0)


def collate_fn(example):
    items = list(zip(*example))

    def merge(data: List[torch.Tensor]):
        max_length = max((t != 0).sum() for t in data)
        # print(f'collate fn : {max_length}')
        return torch.stack([tensor[:max_length].long() for tensor in data], dim=0)

    merge_idxs = [merge(item) for item in items[:4]]

    return merge_idxs + items[4:]


def masked_softmax(x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                   dim: int = -1, log_softmax: bool = False) -> torch.Tensor:
    """
    mask: 1 代表 被 mask 掉的
    """
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    if mask is not None: x = (1 - mask.int()) * x + mask * -1e30
    probs = softmax_fn(x, dim=dim)
    return probs


def beam_search(article: torch.Tensor, article_extend_vocab: torch.Tensor, model: nn.Module,
                id2token: dict, article_oov: list, oov_length: int, args):
    """
    single example
    """
    device = article.device
    mask = article == 0
    lengths = (~mask).sum(dim=-1, keepdim=True)
    length = lengths.item()
    article = article[:length]
    article_extend_vocab = article_extend_vocab[:length].unsqueeze(dim=0)
    mask = mask[:length]
    article_emb = model.emb(article)
    enc_h, dec_h, dec_c = model.encoder(article_emb.unsqueeze(dim=0), lengths)
    coverage_vector = torch.zeros_like(mask.view(1, -1), dtype=torch.float)
    Hypothesis = namedtuple('Hypothesis', ['log_prob', 'y', 'hn', 'cn', "cover_vec", 'dec_seq'])
    start = model.emb(torch.tensor([2], device=device))
    hypothesis = [Hypothesis(log_prob=.0, y=start, hn=dec_h, cn=dec_c,
                             cover_vec=coverage_vector, dec_seq='')]
    length = 0
    start = torch.cat([hypothesis[0].y for _ in range(args.beam_size)], dim=0).to(device)
    hn = torch.cat([hypothesis[0].hn for _ in range(args.beam_size)], dim=0).to(device)
    cn = torch.cat([hypothesis[0].cn for _ in range(args.beam_size)], dim=0).to(device)
    cover_vec = torch.cat([hypothesis[0].cover_vec for _ in range(args.beam_size)], dim=0).to(device)

    results = []
    beam_size = args.beam_size
    while length < args.max_summary_length + 1 and len(results) < args.beam_size:
        print(f'decoding {length + 1} tokens ...')
        candidates = []
        p_extend = torch.zeros((beam_size, oov_length), dtype=torch.float, device=device)
        cur_article_extend_vocab = article_extend_vocab.expand(beam_size, -1)
        cur_enc_h = enc_h.expand(beam_size, -1, -1)
        prob, hn, cn, cover_vec, covloss = model.decoder.step(start, hn, cn, cur_enc_h, cover_vec, mask,
                                                              cur_article_extend_vocab, p_extend, is_training=False)
        topk_prob, topk_idxs = prob.topk(k=beam_size, dim=-1)
        num_hyps = beam_size if length > 0 else 1
        for i in range(num_hyps):
            cur = hypothesis[i]
            for j in range(beam_size):
                log_prob = torch.log(topk_prob[i, j] + 1e-30).item() + cur.log_prob
                idx = topk_idxs[i, j].item()
                if idx < len(id2token):
                    y = model.emb(torch.tensor([idx], device=device))
                    token = id2token[str(idx)]
                else:
                    y = model.emb(torch.tensor([1], device=device))
                    token = article_oov[idx - len(id2token)]
                hyp = Hypothesis(log_prob=log_prob, y=y, hn=hn[[i]], cn=cn[[i]],
                                 cover_vec=cover_vec[[i]], dec_seq=cur.dec_seq + token)
                candidates.append(hyp)
        candidates.sort(key=lambda h: h.log_prob, reverse=True)
        hypothesis = []
        for i in range(beam_size):
            last_token = candidates[i].dec_seq[-5:]
            if last_token == '<eos>':
                results.append(candidates[i])
                beam_size -= 1
            else:
                hypothesis.append(candidates[i])

        if beam_size == 0: break

        start = torch.cat([h.y for h in hypothesis], dim=0).to(device)
        hn = torch.cat([h.hn for h in hypothesis], dim=0).to(device)
        cn = torch.cat([h.cn for h in hypothesis], dim=0).to(device)
        cover_vec = torch.cat([h.cover_vec for h in hypothesis], dim=0).to(device)

        length += 1

    for i in range(beam_size):
        results.append(hypothesis[i])

    results.sort(key=lambda h: h.log_prob, reverse=True)
    result = results[0]
    log_prob, dec_seq = - result.log_prob, result.dec_seq
    if dec_seq[-5:] != '<eos>':
        length = len(dec_seq)
    else:
        dec_seq = dec_seq[:-5]
        length = len(dec_seq) + 1
    loss = log_prob / length
    return loss, dec_seq


def rouge_1(golds: List[str], summary: str):
    summary_counter = Counter(summary)
    numerator = denominator = 0
    for gold in golds:
        gold_counter = Counter(gold)
        common = gold_counter & summary_counter
        numerator += sum(common.values())
        denominator += sum(gold_counter.values())
    return numerator / denominator


def rouge_2(golds: List[str], summary: str):
    bigram = [summary[i:i + 2] for i in range(len(summary) - 1)]
    summary_counter = Counter(bigram)
    numerator = denominator = 0
    for gold in golds:
        gold_bigram = [gold[i:i + 2] for i in range(len(gold) - 1)]
        gold_counter = Counter(gold_bigram)
        common = gold_counter & summary_counter
        numerator += sum(common.values())
        denominator += sum(gold_counter.values())
    return numerator / denominator


def longest_common_subsequence(str1: str, str2: str):
    if not str1 or not str2: return 0
    dp = [[0] * (len(str2) + 1) for _ in range(2)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            row = i % 2
            dp[row][j] = dp[1 - row][j - 1]
            if str1[i - 1] == str2[j - 1]:
                dp[row][j] += 1
            dp[row][j] = max(dp[row][j], dp[row][j - 1], dp[1 - row][j])
    return dp[len(str1) % 2][-1]


def rouge_L(golds: List[str], summary: str):
    fs = []
    for gold in golds:
        lcs = longest_common_subsequence(gold, summary)
        p = lcs / (len(summary))
        r = lcs / (len(gold))
        f = (2 * p * r) / (p + r + 1e-8)
        fs.append(f)
    return sum(fs) / len(fs)


def eval_metrics(golds: List[str], preds: List[str]):
    rouge = Rouge()
    rouge1, rouge2, rougeL = [], [], []
    for gold, pred in zip(golds, preds):
        pred = ' '.join(pred)
        gold = ' '.join(gold)
        res = rouge.get_scores(pred, gold)[0]
        rouge1.append(res['rouge-1']['r'])
        rouge2.append(res['rouge-2']['r'])
        rougeL.append(res['rouge-l']['f'])
    metrics = [
        ('ROUGE-1', 100 * sum(rouge1) / len(rouge1)),
        ('ROUGE-2', 100 * sum(rouge2) / len(rouge2)),
        ('ROUGE-L', 100 * sum(rougeL) / len(rougeL))
    ]
    return metrics


def get_available_device(debug: bool = False):
    gpu_ids = []
    if torch.cuda.is_available() and not debug:
        gpu_ids += [id_ for id_ in range(torch.cuda.device_count())]
        device = torch.device('cuda', 0)
        # torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device, gpu_ids


def tensor_from_json(data_path: str, dtype=torch.float) -> torch.Tensor:
    with open(data_path, 'r') as f:
        data = json.load(f)
    return torch.tensor(data, dtype=dtype)


def set_random(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class ExponentialMovingAverage:

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.origin = {}
        self.shadow = {}
        for name, para in model.named_parameters():
            if para.requires_grad:
                self.shadow[name] = para.data.clone()

    def __call__(self, model: nn.Module, num_updates: int):
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        for name, para in model.named_parameters():
            if para.requires_grad:
                new_average = (1 - decay) * para.data + decay * self.shadow[name]
                self.shadow[name] = new_average

    def assign(self, model: nn.Module):
        for name, para in model.named_parameters():
            if para.requires_grad:
                self.origin[name] = para.data.clone()
                para.data = self.shadow[name]

    def resume(self, model: nn.Module):
        for name, para in model.named_parameters():
            if para.requires_grad:
                para.data = self.origin[name]


class CheckPointSaver:

    def __init__(self, save_dir: str, max_checkpoints: int, metric_name: str,
                 maximize_metric: bool = True, log=None):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.maximize_metric = maximize_metric
        self.metric_name = metric_name
        self.best_val = None
        self.best_path = os.path.join(self.save_dir, 'best.pth.tar')
        self.logger = log
        self.ckpt_paths = []
        self._print(f'Saver will {"max" if maximize_metric else "min"}imize {metric_name} ...')

    def _print(self, message=None):
        if self.logger is not None:
            self.logger.info(message)

    def is_best(self, metric_val: float) -> bool:
        if metric_val is None:
            return False
        if self.best_val is None:
            return True
        return self.best_val < metric_val and self.maximize_metric \
               or self.best_val > metric_val and not self.maximize_metric

    def save(self, model: nn.Module, optimizer, metric_val: float, steps: int, device: torch.device):
        ckpt_path = os.path.join(self.save_dir, f'steps_{steps}.pth.tar')
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'steps': steps
        }
        model.to(device)
        priority_val = metric_val if self.maximize_metric else - metric_val
        if len(self.ckpt_paths) < self.max_checkpoints:
            heapq.heappush(self.ckpt_paths, (priority_val, ckpt_path))
            torch.save(ckpt_dict, ckpt_path)
            self._print(f'Saved CheckPoint: {ckpt_path}')
        else:
            tmp_val, tmp_path = heapq.heappushpop(self.ckpt_paths, (priority_val, ckpt_path))
            if tmp_val != priority_val:
                torch.save(ckpt_dict, ckpt_path)
                self._print(f'Saved CheckPoint: {ckpt_path}')
                if self.is_best(metric_val):
                    self.best_val = metric_val
                    shutil.copy(ckpt_path, self.best_path)
                    self._print(f'New best CheckPoint at {steps} ...')
                try:
                    os.remove(tmp_path)
                    self._print(f'Removed CheckPoint: {tmp_path} ...')
                except OSError:
                    pass


def get_logger(log_dir):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_save_dir(base: str, training: bool, id_max: int = 100):
    subdir = 'train' if training else 'test'
    for uid in range(1, id_max):
        save_dir = os.path.join(base, f'{subdir}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir
    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


class AverageMeter:

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def reset(self):
        self.__init__()

    def update(self, val: float, num_examples: int):
        self.count += num_examples
        self.sum += val * num_examples
        self.avg = self.sum / self.count


def visualize(tbx: SummaryWriter, articles: List[str], golds: List[str],
              preds: List[str], num_visuals: int, split: str, steps: int):
    if num_visuals < 0:
        return

    num_visuals = min(num_visuals, len(golds))
    visual_ids = np.random.choice(range(len(golds)), num_visuals, replace=False)

    for i, id_ in enumerate(visual_ids):
        pred = preds[id_] or 'N/A'
        gold = golds[id_]
        article = articles[id_]

        tbl_fmt = f'- **原文本** : {article}\n' \
                  f'- **参考摘要** : {gold}\n' \
                  f'- **预测摘要** : {pred}'

        tbx.add_text(f'{split}/{i + 1}-{num_visuals}',
                     tbl_fmt, steps)


def load_model(model: nn.Module, model_path: str, gpus: list, return_step: bool = True):
    device = torch.device(f'cuda: {gpus[0]}' if gpus else 'cpu')
    ckpt_state = torch.load(model_path, map_location=device)

    model.load_state_dict(ckpt_state['model_state'])

    if return_step:
        return model, ckpt_state['steps']

    return model


if __name__ == '__main__':
    a = '福州维修店“偷”走行货零件重组手机获利'
    aa = ' '.join(a)
    b = '苹果手机屏幕被老板私自“劫”<eos>'
    bb = ' '.join(b)
    rouge = Rouge()
    res = rouge.get_scores([bb], [aa])
    print(dumps(res, indent=4))
    print(rouge_1([a], b))
    print(rouge_2([a], b))
    print(rouge_L([a], b))
