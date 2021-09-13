# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 3:47 下午
# @Author  : zhengjiawei
# @FileName: predict.py
# @Software: PyCharm
import json
import os

import pandas as pd
import torch
from rouge import Rouge

from src.build_data import load_dataset
from src.seq2seq_torch.predict_help import beam_decode, greedy_decode
from src.seq2seq_torch.seq2seq_batcher import beam_test_batch_generator
from src.seq2seq_torch.seq2seq_model import Seq2Seq
from src.utils.config import seq2seq_checkpoint_dir, test_x_path, test_y_path, test_seg_path
from src.utils.gpu_utils import config_gpu
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    config_gpu()
    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count
    print("Building the model ...")
    model = Seq2Seq(params, vocab).to(params['device'])

    checkpoint = torch.load(seq2seq_checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if params['greedy_decode']:
        print('Using greedy search to decoding ...')
        predict_result(model, params, vocab)
    else:
        print('Using beam search to decoding ...')
        b = beam_test_batch_generator(params["beam_size"])
        results = []
        for batch in b:
            batch = batch.cuda()
            best_hyp = beam_decode(model, batch, vocab, params)
            results.append(best_hyp.abstract)
        get_rouge(results)
        print('save result to :{}'.format(params['result_save_path']))


def predict_result(model, params, vocab):
    test_x, _ = load_dataset(test_x_path, test_y_path,
                             params['max_enc_len'], params['max_dec_len'])
    # 预测结果
    results = greedy_decode(model, test_x, params['batch_size'], vocab, params)
    # 保存结果
    get_rouge(results)


def get_rouge(results):
    # 读取结果
    seg_test_report = pd.read_csv(test_seg_path, header=None).iloc[:, 5].tolist()
    seg_test_report = [' '.join(str(token) for token in str(line).split()) for line in seg_test_report]
    rouge_scores = Rouge().get_scores(results, seg_test_report, avg=True)
    print_rouge = json.dumps(rouge_scores, indent=2)
    with open(os.path.join(os.path.dirname(test_seg_path), 'results.csv'), 'w', encoding='utf8') as f:
        json.dump(list(zip(results, seg_test_report)), f, indent=2, ensure_ascii=False)
    print('*' * 8 + ' rouge score ' + '*' * 8)
    print(print_rouge)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    params['greedy_decode'] = False
    params['beam_size'] = params['batch_size'] = 4
    # 获得参数
    test(params)
