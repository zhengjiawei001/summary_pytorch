import json

import numpy as np
import pandas as pd
import torch
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pgn_torch.dataset import PairDataset, SampleDataset, collate_fn
from src.pgn_torch.layers import PGN
# from src.utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs
from src.pgn_torch.utils import replace_oovs
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    if params['decode_mode'] == 'beam':
        assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    model = PGN(params)
    model.load_state_dict(torch.load(os.path.join(params['checkpoint_dir'], "best_model.pkl")))

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    results = predict_result(model, params, vocab, params['result_save_path'])

    print('save result to :{}'.format(params['result_save_path']))
    print('save result :{}'.format(results[:5]))


def predict_result(model, params, vocab, result_save_path):
    # 获得dataset
    test_data = SampleDataset(params, 'test', vocab, PairDataset)
    test_iter = DataLoader(dataset=test_data,
                           batch_size=params['batch_size'],
                           shuffle=True,
                           collate_fn=collate_fn)
    if params['decode_mode'] == 'beam':
        print('1111')
        results = []
        for batch in tqdm(test_iter):
            best_hyp = beam_decode(model, batch, vocab, params)
            results.append(best_hyp.abstract)

    else:
        print('2222')
        results = greedy_decode(model, test_iter, vocab, params)
    get_rouge(results, params)

    # 保存结果
    if not os.path.exists(os.path.dirname(result_save_path)):
        os.makedirs(os.path.dirname(result_save_path))
    # save_predict_result(results,result_save_path)

    return results


# def save_predict_result(results, result_save_path):
#     # 读取结果
#     test_df = pd.read_csv(test_data_path)
#     # 填充结果
#     test_df['Prediction'] = results[:len(test_df['QID'])]
#     # 提取ID和预测结果两列
#     test_df = test_df[['QID', 'Prediction']]
#     # 保存结果.
#     test_df.to_csv(result_save_path, index=None, sep=',')

def get_rouge(results, params):
    # 读取结果
    seg_test_report = pd.read_csv(params['test_data_path'], header=None).iloc[:len(results), 5].tolist()
    seg_test_report = [' '.join(str(token) for token in str(line).split()) for line in seg_test_report]
    print('seg_test_report:', len(seg_test_report))
    print('results:', len(results))
    rouge_scores = Rouge().get_scores(results, seg_test_report, avg=True)
    print_rouge = json.dumps(rouge_scores, indent=2)
    print('*' * 8 + ' rouge score ' + '*' * 8)
    print(print_rouge)


def greedy_decode(model, dataset, vocab, params):
    results = []
    # sample_size = 20000
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    # steps_epoch = sample_size // batch_size + 1
    # [0,steps_epoch)
    ds = iter(dataset)
    for i in tqdm(range(50)):
        enc_data = next(ds)
        batch_results = batch_greedy_decode(model, enc_data, vocab, params)
        results.extend(batch_results)
    return results


def batch_greedy_decode(model, encoder_batch_data, vocab, params):
    # 判断输入长度
    x, y, x_len, y_len, oov, len_oovs = encoder_batch_data
    batch_size = x.shape[0]

    # article_oovs = encoder_batch_data['article_oovs'].numpy()

    # 开辟结果存储list
    predicts = [''] * batch_size
    enc_input = x

    # encoder
    enc_output, enc_hidden = model.encoder(enc_input)

    decoder_states = enc_hidden

    dec_input = torch.tensor([vocab.START_DECODING_INDEX] * batch_size)

    enc_input = replace_oovs(enc_input, vocab)

    # torch.ne 计算是否不等于，x中的元素不等于0 返回True,等于返回False
    enc_input_padding_masks = torch.ne(enc_input, 0).float()

    coverage_vector = torch.zeros((1, enc_input.shape[1])).to(params['device'])
    for t in range(params['max_dec_len']):
        # 单步预测
        context_vector, attention_weights, coverage_vector = model.attention(decoder_states[0],
                                                                             enc_output,
                                                                             enc_input_padding_masks,
                                                                             use_coverage=True,
                                                                             prev_coverage=coverage_vector)
        p_vocab, decoder_states, p_gen = model.decoder(dec_input.unsqueeze(1),
                                                       decoder_states,
                                                       context_vector)
        final_dist = model.get_final_distribution(enc_input,
                                                  p_gen,
                                                  p_vocab,
                                                  attention_weights,
                                                  torch.max(len_oovs))
        # Get next token with maximum probability.
        predicted_ids = torch.argmax(final_dist, dim=1)

        inp_predicts = []
        for index, predicted_id in enumerate(predicted_ids.numpy()):
            if predicted_id >= vocab.count:
                # OOV词
                word = oov[index][predicted_id - vocab.count].decode()
                inp_predicts.append(vocab.UNKNOWN_TOKEN_INDEX)
            else:
                word = vocab.id_to_word(predicted_id)
                inp_predicts.append(predicted_id)
            predicts[index] += word + ' '

        predicted_ids = torch.tensor(inp_predicts)
        # using teacher forcing
        dec_input = predicted_ids

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if vocab.STOP_DECODING in predict:
            # 截断stop
            predict = predict[:predict.index(vocab.STOP_DECODING)]
        # 保存结果
        results.append(predict)
    return results


def print_top_k(hyp, k, vocab, batch):
    text = batch[0]["article"].numpy()[0].decode()
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    print('\nhyp.text :{}'.format(text))
    for i in range(min(k, len(hyp))):
        k_hyp = hyp[i]
        k_hyp = result_index2text(k_hyp, vocab, batch)
        print('top {} best_hyp.abstract :{}\n'.format(i, k_hyp.abstract))


def result_index2text(hyp, vocab, batch):
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    hyp.article = batch[0]["article"].numpy()[0].decode()

    words = []
    for index in hyp.tokens:
        if index != vocab.START_DECODING_INDEX and index != vocab.STOP_DECODING_INDEX:
            if index < (len(article_oovs) + vocab.size()):
                if index < vocab.size():
                    words.append(vocab.id_to_word(index))
                else:
                    words.append(article_oovs[index - vocab.size()].decode())
            else:
                print('error values id :{}'.format(index))
    hyp.abstract = " ".join(words)
    return hyp


def beam_decode(model, batch, vocab, params, print_info=False):
    # 初始化mask
    start_index = vocab.word_to_id(vocab.START_DECODING)
    stop_index = vocab.word_to_id(vocab.STOP_DECODING)
    unk_index = vocab.word_to_id(vocab.UNKNOWN_TOKEN)
    batch_size = 1

    x, y, x_len, y_len, oov, len_oovs = batch
    enc_input = x
    enc_extended_inp = replace_oovs(enc_input, vocab)
    # torch.ne 计算是否不等于，x中的元素不等于0 返回True,等于返回False
    enc_input_padding_masks = torch.ne(enc_input, 0).float()

    def decoder_one_step(encoder_output, decoder_input, decoder_hidden, enc_input,
                         batch_oov_length, encoder_pad_mask,
                         use_coverage, coverage_vector):
        # 单个时间步，运行
        context_vector, attention_weights, coverage_vector1 = model.attention(decoder_hidden[0],
                                                                              encoder_output,
                                                                              encoder_pad_mask,
                                                                              use_coverage=use_coverage,
                                                                              prev_coverage=coverage_vector)

        print('coverage_vector1:', coverage_vector1.size())
        p_vocab, decoder_states, p_gen = model.decoder(decoder_input.unsqueeze(1),
                                                       decoder_hidden,
                                                       coverage_vector1)
        final_dist = model.get_final_distribution(enc_input,
                                                  p_gen,
                                                  p_vocab,
                                                  attention_weights,
                                                  torch.max(batch_oov_length))

        top_k_probs, top_k_ids = torch.topk(final_dist.squeeze(), k=params["beam_size"] * 2)

        results = {
            'coverage_ret': coverage_vector1,
            "last_context_vector": context_vector,
            "dec_hidden": dec_hidden,
            "attention_weights": attention_weights,
            "top_k_ids": top_k_ids,
            "top_k_log_probs": top_k_probs,
            "p_gens": p_gens}

        # 返回需要保存的中间结果和概率
        return results

    # 测试数据的输入
    # x, y, x_len, y_len, oov, len_oovs
    # enc_pad_mask = batch[0]["enc_pad_mask"]
    batch_oov_len = len_oovs
    # enc_pad_mask = batch[0]["encoder_pad_mask"]
    # article_oovs = oov
    enc_output, enc_hidden = model.encoder(enc_input)

    # 初始化batch size个 假设对象
    hyps = [Hypothesis(tokens=[start_index],
                       log_probs=[0.0],
                       hidden=enc_hidden,
                       attn_dists=[],
                       p_gens=[],
                       # zero vector of length attention_length
                       coverage=np.zeros([enc_input.shape[1], 1], dtype=np.float32)) for _ in range(batch_size)]

    # 初始化结果集
    results = []  # list to hold the top beam_size hypothesises
    # 遍历步数
    steps = 0  # initial step

    # 长度还不够 并且 结果还不够 继续搜索
    while steps < params['max_dec_len'] and len(results) < params['beam_size']:
        # 获取最新待使用的token
        latest_tokens = [h.latest_token for h in hyps]
        # 替换掉 oov token unknown token
        latest_tokens = [t if t in vocab.id2word else unk_index for t in latest_tokens]

        # 获取所以隐藏层状态
        hidden = [h.hidden for h in hyps]
        coverage_vector = [h.coverage for h in hyps]
        prev_coverage = torch.tensor(coverage_vector).squeeze(-1)

        print(len(hidden[0]))
        print(hidden[0][0].size())
        dec_hidden = torch.stack(hidden, dim=0)

        # print('hidden_len:', len(dec_hidden))
        print('hidden_size:', dec_hidden.size())

        dec_input = torch.tensor(latest_tokens)

        decoder_results = decoder_one_step(enc_output,
                                           dec_input,
                                           dec_hidden,
                                           enc_extended_inp,
                                           batch_oov_len,
                                           enc_input_padding_masks,
                                           use_coverage=True,
                                           coverage_vector=prev_coverage)

        dec_hidden = decoder_results['dec_hidden']
        attention_weights = decoder_results['attention_weights']
        top_k_log_probs = decoder_results['top_k_log_probs']
        top_k_ids = decoder_results['top_k_ids']
        new_coverage = decoder_results['coverage_ret']
        p_gens = decoder_results['p_gens']

        # 现阶段全部可能情况
        all_hyps = []
        # 原有的可能情况数量 TODO
        num_orig_hyps = 1 if steps == 0 else len(hyps)

        # 遍历添加所有可能结果
        for i in range(num_orig_hyps):
            h, new_hidden, attn_dist = hyps[i], dec_hidden[i], attention_weights[i]
            if params['pointer_gen']:
                p_gen = p_gens[i]
            else:
                p_gen = 0
            if params['use_coverage']:
                new_coverage_i = new_coverage[i]
            else:
                new_coverage_i = 0

            # 分裂 添加 beam size 种可能性
            for j in range(params['beam_size'] * 2):
                # 构造可能的情况
                new_hyp = h.extend(token=top_k_ids[i, j].numpy(),
                                   log_prob=top_k_log_probs[i, j],
                                   hidden=new_hidden,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                # 添加可能情况
                all_hyps.append(new_hyp)

        # 重置
        hyps = []
        # 按照概率来排序
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)

        # 筛选top前beam_size句话
        for h in sorted_hyps:
            if h.latest_token == stop_index:
                # 长度符合预期,遇到句尾,添加到结果集
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                # 未到结束 ,添加到假设集
                hyps.append(h)

            # 如果假设句子正好等于beam_size 或者结果集正好等于beam_size 就不在添加
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break

        steps += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)

    best_hyp = hyps_sorted[0]
    # best_hyp.abstract = " ".join([vocab.id_to_word(index) for index in best_hyp.tokens])
    # best_hyp.text = batch[0]["article"].numpy()[0].decode()

    best_hyp = result_index2text(best_hyp, vocab, batch)

    if print_info:
        print_top_k(hyps_sorted, params['beam_size'], vocab, batch)
        print('real_article: {}'.format(best_hyp.real_abstract))
        print('article: {}'.format(best_hyp.abstract))

    best_hyp = result_index2text(best_hyp, vocab, batch)
    return best_hyp


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, hidden, attn_dists, p_gens, coverage):
        self.tokens = tokens  # list of all the tokens from time 0 to the current time step t
        self.log_probs = log_probs  # list of the log probabilities of the tokens of the tokens
        self.hidden = hidden  # decoder hidden state after the last token decoding
        self.attn_dists = attn_dists  # attention dists of all the tokens
        self.p_gens = p_gens
        self.coverage = coverage
        self.abstract = ""
        self.real_abstract = ""
        self.article = ""

    def extend(self, token, log_prob, hidden, attn_dist, p_gen, coverage):
        """Method to extend the current hypothesis by adding the next decoded token and all the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          hidden=hidden,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


if __name__ == '__main__':
    import os

    # 获得参数
    params = get_params()

    # beam search
    params['batch_size'] = 4
    params['beam_size'] = 4
    params['mode'] = 'test'
    params['decode_mode'] = 'beam'
    params['pointer_gen'] = True
    params['use_coverage'] = True
    params['enc_units'] = 128
    params['dec_units'] = 256
    params['attn_units'] = 128
    params['min_dec_steps'] = 3
    params['max_enc_len'] = 200
    params['max_dec_len'] = 40

    # greedy search
    # params['batch_size'] = 8
    # params['mode'] = 'test'
    # params['decode_mode'] = 'greedy'
    # params['pointer_gen'] = True
    # params['use_coverage'] = True
    # params['enc_units'] = 256
    # params['dec_units'] = 512
    # params['attn_units'] = 256
    # params['min_dec_steps'] = 3

    test(params)
