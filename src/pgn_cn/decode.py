# run(['pip', 'install', 'ujson'])
# run(['pip', 'install', 'rouge'])

from collections import OrderedDict
from json import dumps

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

from src.pgn_cn import utils
from src.pgn_cn.args import test_args
from src.pgn_cn.model import Summary


def decode(args):
    args.save_dir = utils.get_save_dir(args.save_dir, True)
    logger = utils.get_logger(args.save_dir)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpus = utils.get_available_device()

    with open(args.pretrain_token2id, 'r', encoding='utf8') as f1, \
            open(args.pretrain_id2token, 'r', encoding='utf8') as f2:
        token2id = json_load(f1)
        id2token = json_load(f2)

    args.vocab_size = len(token2id)
    logger.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')

    logger.info(f'Using random seed {args.seed}')
    utils.set_random(args.seed)

    logger.info('Building model ...')
    weight = torch.load(args.pretrain_embedding_file)
    assert weight.size(0) == args.vocab_size, (weight.size(0), args.vocab_size)
    model = Summary(weight=weight, embedding_dim=args.embedding_dim,
                    hidden_size=args.hidden_size, vocab_size=args.vocab_size)

    new_state_dict = OrderedDict()
    state = torch.load(args.best_model)
    model_state, steps = state['model_state'], state['steps']
    for k, v in model_state.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

    testset = utils.LCSTS(args.pretrain_test_idxs, is_training=False)

    with torch.no_grad():
        model.eval()
        losses = []
        dec_seqs, golds, origin_articles = [], [], []
        for i, (article_idx, article_extend_idx, _, _, _, summary, article, article_oov) in tqdm(enumerate(testset)):
            article_idx, article_extend_idx = article_idx.to(device), \
                                              article_extend_idx.to(device)
            golds.append(summary)
            origin_articles.append(article)
            loss, dec_seq = utils.beam_search(article_idx, article_extend_idx, model,
                                              id2token, article_oov, len(article_oov), args)

            # if np.random.random() < args.visiual_prob:
            tbl_fmt = f'- **原文本** ： {article}\n' \
                      f'- **参考摘要** : {summary}\n' \
                      f'- **预测摘要** : {dec_seq}'
            tbx.add_text(f'num_visuals-{i + 1}', tbl_fmt, steps)

            losses.append(loss)
            dec_seqs.append(dec_seq)

        metric_results = utils.eval_metrics(golds, dec_seqs)
        metric_results.append(('NLL', sum(losses) / len(losses)))
        metric_results = OrderedDict(metric_results)

    result_str = ', '.join(f'{metric_name} : {value:05.2f}' for metric_name, value in metric_results.items())
    logger.info(f'final results: {result_str}')


if __name__ == '__main__':
    args = test_args()
    decode(args)
