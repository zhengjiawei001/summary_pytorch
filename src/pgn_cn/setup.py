# -*- coding : utf-8 -*-

import bz2
import re
from collections import Counter

import numpy as np
import torch
import ujson as json
from tqdm import tqdm

from src.pgn_cn.args import setup_args


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w", encoding='utf8') as fh:
            json.dump(obj, fh)


def get_embedding(args, word_counter: dict, min_df: int = -1) -> dict:
    print('processing embedding file ...')

    filter_items = [word for word, freq in word_counter.items() if freq > min_df]
    token2embedding = {}

    with bz2.open(args.pretrain_file) as f:

        token_vectors = f.readlines()
        meta_info = token_vectors[0].split()
        print(f'{meta_info[0]} tokens in embedding file in total, vector size is {meta_info[-1]}')

        for line in tqdm(token_vectors[1:]):
            line = line.split()
            token = line[0].decode('utf8')
            vector = line[1:]
            if token in word_counter and word_counter[token] > min_df:
                token2embedding[token] = [float(num) for num in vector]

        print(f'{len(token2embedding)}/{len(filter_items)} tokens have corresponding embedding vectors')

    token2idx = {token: idx for idx, token in enumerate(token2embedding.keys(), 4)}
    UNK, PAD, BOS, EOS = '<unk>', '<pad>', '<bos>', '<eos>'
    token2idx[PAD] = args.pad
    token2idx[UNK] = args.unk
    token2idx[BOS] = args.bos
    token2idx[EOS] = args.eos
    idx2token = {idx: token for token, idx in token2idx.items()}
    idx2embedding = {token2idx[token]: embedding for token, embedding in token2embedding.items()}
    idx2embedding[args.pad] = [.0] * int(meta_info[-1])
    idx2embedding[args.unk] = [.0] * int(meta_info[-1])
    idx2embedding[args.bos] = np.random.random(int(meta_info[-1])).tolist()
    idx2embedding[args.eos] = np.random.random(int(meta_info[-1])).tolist()
    emb_mat = [idx2embedding[idx] for idx in range(len(idx2embedding))]

    save(args.pretrain_token2id, token2idx, message='token2id')
    save(args.pretrain_id2token, idx2token, message='id2token')
    torch.save(torch.tensor(emb_mat, dtype=torch.float), args.pretrain_embedding_file)

    return token2idx


def build_idxs(token2id: dict, pairs: list, args):
    article_oovs = []
    article_extend_idxs = []
    article_idxs = []
    summary_extend_idxs = []
    summary_idxs = []
    for article, summary in tqdm(pairs):
        article_extend_idx = [args.pad] * args.max_article_length
        article_idx = [args.pad] * args.max_article_length
        summary_extend_idx = [args.pad] * (args.max_summary_length + 1)
        summary_idx = [args.pad] * (args.max_summary_length + 1)
        summary_idx[0] = args.bos
        article_oov = []
        for i, token in enumerate(article):
            article_idx[i] = token2id.get(token, args.unk)
            if token not in token2id:
                if token not in article_oov:
                    article_oov.append(token)
                idx = article_oov.index(token)
                idx += len(token2id)
            else:
                idx = token2id[token]
            article_extend_idx[i] = idx

        for j, token in enumerate(summary):
            summary_idx[j + 1] = token2id.get(token, args.unk)
            if token not in token2id:
                if token in article_oov:
                    idx = article_oov.index(token) + len(token2id)
                else:
                    idx = args.unk
            else:
                idx = token2id[token]
            summary_extend_idx[j] = idx
        summary_extend_idx[len(summary)] = args.eos

        article_oovs.append(article_oov)
        article_idxs.append(article_idx)
        article_extend_idxs.append(article_extend_idx)
        summary_idxs.append(summary_idx)
        summary_extend_idxs.append(summary_extend_idx)

    return article_oovs, article_idxs, article_extend_idxs, \
           summary_idxs, summary_extend_idxs


def process_training_file(train_file_path: str, args, debug=True):
    print('Starting process training file ...')
    article_pattern = re.compile(r'<short_text>(.*?)</short_text>', flags=re.S)
    summary_pattern = re.compile(r'<summary>(.*?)</summary>', flags=re.S)
    with open(train_file_path, 'r', encoding='utf8') as f:
        content = f.read()
        if debug: content = content[:1000]
        articles = [t.strip() for t in article_pattern.findall(content)]
        summaries = [s.strip() for s in summary_pattern.findall(content)]
    if debug: summaries = summaries[:-1]
    assert len(articles) == len(summaries), (len(articles), len(summaries))

    pairs = list(zip(articles, summaries))
    word_counter = Counter(''.join(article + summary for article, summary in pairs))

    token2id = get_embedding(args, word_counter)
    article_oovs, article_idxs, article_extend_idxs, \
    summary_idxs, summary_extend_idxs = build_idxs(token2id, pairs, args)

    article_idxs = torch.tensor(article_idxs, dtype=torch.int16)
    article_extend_idxs = torch.tensor(article_extend_idxs, dtype=torch.int16)
    summary_idxs = torch.tensor(summary_idxs, dtype=torch.int16)
    summary_extend_idxs = torch.tensor(summary_extend_idxs, dtype=torch.int16)

    print(f'article index : {article_idxs.size()}')
    print(f'article extend index : {article_extend_idxs.size()}')
    print(f'summary index : {summary_idxs.size()}')
    print(f'summary extend index : {summary_extend_idxs.size()}')

    idx = np.arange(article_extend_idxs.size(0))
    np.random.shuffle(idx)
    summaries = [summaries[i] for i in idx]
    articles = [articles[i] for i in idx]
    article_oovs = [article_oovs[i] for i in idx]
    idx = torch.from_numpy(idx).long()
    article_idxs = article_idxs[idx]
    article_extend_idxs = article_extend_idxs[idx]
    summary_idx = summary_idxs[idx]
    summary_extend_idx = summary_extend_idxs[idx]

    split = int(article_extend_idxs.size(0) * args.valid_split)
    valid_summaries = summaries[:split]
    valid_articles = articles[:split]
    valid_article_oovs = article_oovs[:split]
    valid_article_idxs = article_idxs[:split]
    valid_article_extend_idxs = article_extend_idxs[:split]
    valid_summary_idxs = summary_idx[:split]
    valid_summary_extend_idxs = summary_extend_idx[:split]

    oovs_length = [len(oov) for oov in article_oovs[split:]]
    train_article_idxs = article_idxs[split:]
    train_article_extend_idxs = article_extend_idxs[split:]
    train_summary_idxs = summary_idx[split:]
    train_summary_extend_idxs = summary_extend_idx[split:]

    print('Saving training idx ...')
    torch.save(
        {
            'article_idxs': train_article_idxs,
            'article_extend_idxs': train_article_extend_idxs,
            'summary_idxs': train_summary_idxs,
            'summary_extend_idxs': train_summary_extend_idxs,
            'oovs_length': oovs_length
        },
        args.pretrain_train_idxs
    )
    print(f'total {train_article_idxs.size(0)} training examples')
    print('Saving valid idx ...')
    torch.save(
        {
            'article_idxs': valid_article_idxs,
            'article_extend_idxs': valid_article_extend_idxs,
            'summary_idxs': valid_summary_idxs,
            'summary_extend_idxs': valid_summary_extend_idxs,
            'summaries': valid_summaries,
            'articles': valid_articles,
            'article_oovs': valid_article_oovs
        },
        args.pretrain_valid_idxs
    )
    print(f'total {valid_article_idxs.size(0)} valid examples')

    return token2id


def process_test_file(test_file_path: str, token2id: dict, args):
    print('Starting process test file ...')
    label_pattern = re.compile(r'<human_label>.*?(\d).*?</human_label>', flags=re.S)
    article_pattern = re.compile(r'<short_text>(.*?)</short_text>', flags=re.S)
    summary_pattern = re.compile(r'<summary>(.*?)</summary>', flags=re.S)
    with open(test_file_path, 'r', encoding='utf8') as f:
        content = f.read()
        labels = [l.strip() for l in label_pattern.findall(content)]
        articles = [t.strip() for t in article_pattern.findall(content)]
        summaries = [s.strip() for s in summary_pattern.findall(content)]

    assert len(articles) == len(summaries) == len(labels)
    pairs = [(t, s) for l, t, s in zip(labels, articles, summaries) if int(l) >= 3]
    print(f'origin {len(labels)} / filter {len(pairs)} ...')

    article_oovs, article_idxs, article_extend_idxs, \
    summary_idxs, summary_extend_idxs = build_idxs(token2id, pairs, args)

    article_idxs = torch.tensor(article_idxs, dtype=torch.int16)
    article_extend_idxs = torch.tensor(article_extend_idxs, dtype=torch.int16)
    summary_idxs = torch.tensor(summary_idxs, dtype=torch.int16)
    summary_extend_idxs = torch.tensor(summary_extend_idxs, dtype=torch.int16)

    print(f'article index : {article_idxs.size()}')
    print(f'article extend index : {article_extend_idxs.size()}')
    print(f'summary index : {summary_idxs.size()}')
    print(f'summary extend index : {summary_extend_idxs.size()}')

    print('Saving test idx ...')
    torch.save(
        {
            'article_idxs': article_idxs,
            'article_extend_idxs': article_extend_idxs,
            'summary_idxs': summary_idxs,
            'summary_extend_idxs': summary_extend_idxs,
            'summaries': summaries,
            'articles': articles,
            'article_oovs': article_oovs
        },
        args.pretrain_test_idxs
    )


if __name__ == '__main__':
    args = setup_args()
    token2id = process_training_file(args.train_file, args, debug=False)
    process_test_file(args.test_file, token2id, args)
