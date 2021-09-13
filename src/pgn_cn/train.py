# from subprocess import run
#
# run(['pip', 'install', 'ujson'])
# run(['pip', 'install', 'rouge'])

from collections import OrderedDict
from json import dumps

import torch
import torch.nn as nn
import torch.optim as opt
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from ujson import load as json_load

from src.pgn_cn import utils
from src.pgn_cn.args import train_args
from src.pgn_cn.model import Summary


def main(args):
    args.save_dir = utils.get_save_dir(args.save_dir, True)
    logger = utils.get_logger(args.save_dir)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpus = utils.get_available_device()

    with open(args.pretrain_token2id, 'r', encoding='utf8') as f:
        token2id = json_load(f)

    args.vocab_size = len(token2id)
    logger.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(len(args.gpus), 1)

    logger.info(f'Using random seed {args.seed}')
    utils.set_random(args.seed)

    logger.info('Building model ...')
    weight = torch.load(args.pretrain_embedding_file)
    # assert weight.size(0) == args.vocab_size, (weight.size(0), args.vocab_size)
    model = Summary(weight=weight, embedding_dim=args.embedding_dim,
                    hidden_size=args.hidden_size, vocab_size=args.vocab_size)
    if len(args.gpus) != 0: model = nn.DataParallel(model, device_ids=args.gpus)
    if args.load_path:
        logger.info(f'Loading checkpoint from {args.load_path} ...')
        model, steps = utils.load_model(model, args.load_path, args.gpus)
    else:
        steps = 0
    model.to(device)

    ema = utils.ExponentialMovingAverage(model, args.decay)
    saver = utils.CheckPointSaver(args.save_dir, args.max_checkpoints, args.metric_name,
                                  maximize_metric=False, log=logger)

    optimizer = opt.Adagrad(model.parameters(), lr=args.lr,
                            initial_accumulator_value=args.init_accumulator_val)
    if args.load_path:
        optimizer.load_state_dict(torch.load(args.load_path, map_location=device)['optimizer_state'])

    logger.info('Building Dataset ...')
    trainset = utils.LCSTS(args.pretrain_train_idxs)
    validset = utils.LCSTS(args.pretrain_valid_idxs, is_training=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=utils.collate_fn)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=utils.collate_fn)

    logger.info('Starting training ...')
    epoch = steps // len(trainset)
    steps_till_eval = args.eval_steps
    while epoch < args.num_epochs:
        epoch += 1
        logger.info(f'Starting epoch {epoch} ...')
        model.train()
        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for article_idxs, article_extend_idxs, summary_idxs, \
                summary_extend_idxs, oovs_length in train_loader:
                bs = len(oovs_length)
                article_idxs, summary_idxs = article_idxs.to(device), summary_idxs.to(device)
                article_extend_idxs = article_extend_idxs.to(device)
                summary_extend_idxs = summary_extend_idxs.to(device)
                losses = model(article_idxs, article_extend_idxs, summary_idxs, summary_extend_idxs,
                               args.lambda_, args.is_coverage, max(oovs_length))
                loss = losses[0].sum() / bs
                if args.is_coverage:
                    nll_loss = (losses[1].sum() / bs).item()
                    coverage_loss = (losses[2].sum() / bs).item()
                loss_val = loss.item()  # 由于损失已经在模型里计算完毕，如果是多卡这里就会有多个值需要平均

                model.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                ema(model, steps // bs)

                steps += bs
                progress_bar.update(bs)
                progress_bar.set_postfix(NLL=loss_val, epoch=epoch)
                tbx.add_scalar('train/Total-Loss', loss_val, steps)
                if args.is_coverage:
                    tbx.add_scalar('train/NLL', nll_loss, steps)
                    tbx.add_scalar('train/Coverage-Loss', coverage_loss, steps)
                tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], steps)

                steps_till_eval -= bs
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    logger.info(f'Evaluating at steps {steps}...')
                    ema.assign(model)
                    valid_loss = evaluate_only_loss(model, valid_loader, device, args)
                    saver.save(model, optimizer, valid_loss[0], steps, device)
                    ema.resume(model)

                    logger.info(f'Dev: Valid loss : {valid_loss}')

                    logger.info('Visualizing in TensorBoard ...')
                    tbx.add_scalar(f'dev/Total-Loss', valid_loss[0], steps)
                    tbx.add_scalar(f'dev/NLL', valid_loss[1], steps)
                    tbx.add_scalar(f'dev/Coverage-Loss', valid_loss[2], steps)


def evaluate_only_loss(model: nn.Module, dataloader: DataLoader, device: torch.device, args):
    total_loss_meter = utils.AverageMeter()
    nll_meter = utils.AverageMeter()
    cover_loss_meter = utils.AverageMeter()

    model.eval()
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for article_idxs, article_extend_idxs, summary_idxs, \
            summary_extend_idxs, oovs_length, _, _, _ in dataloader:
            bs = len(oovs_length)
            article_idxs, summary_idxs = article_idxs.to(device), summary_idxs.to(device)
            article_extend_idxs = article_extend_idxs.to(device)
            summary_extend_idxs = summary_extend_idxs.to(device)
            losses = model(article_idxs, article_extend_idxs, summary_idxs, summary_extend_idxs,
                           args.lambda_, args.is_coverage, max(oovs_length))
            loss = losses[0].sum() / bs
            if args.is_coverage:
                nll_loss = (losses[1].sum() / bs).item()
                coverage_loss = (losses[2].sum() / bs).item()
            loss_val = loss.item()

            total_loss_meter.update(loss_val, bs)
            if args.is_coverage:
                nll_meter.update(nll_loss, bs)
                cover_loss_meter.update(coverage_loss, bs)
            progress_bar.update(bs)
            progress_bar.set_postfix(NLL=nll_meter.avg)

    model.train()

    results = (total_loss_meter.avg,)
    if args.is_coverage:
        results += (nll_meter.avg, cover_loss_meter.avg)
    return results


def evaluate(model: nn.Module, dataloader: DataLoader, id2token: dict, args, device: torch.device):
    nll_meter = utils.AverageMeter()

    model.eval()
    model.to('cpu')
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        dec_seqs = []
        gold_summaries = []
        origin_articles = []
        for article_idxs, article_extend_idxs, _, _, _, \
            summaries, articles, article_oovs in dataloader:
            losses = []
            gold_summaries.extend(summaries)
            origin_articles.extend(articles)
            for article_idx, article_extend_idx, article_oov in zip(articles, article_idxs, article_oovs):
                loss, dec_seq = utils.beam_search(article_idxs, article_extend_idxs, model,
                                                  id2token, article_oov, len(article_oov), args)
                losses.append(loss)
                dec_seqs.append(dec_seq)

            bs = len(losses)
            nll_meter.update(sum(losses) / bs, bs)
            progress_bar.update(bs)
            progress_bar.set_postfix(NLL=nll_meter.avg)

    model.to(device)
    model.train()

    metric_results = utils.eval_metrics(gold_summaries, dec_seqs)
    metric_results.append(('NLL', nll_meter.avg))

    return OrderedDict(metric_results), gold_summaries, dec_seqs, origin_articles


if __name__ == '__main__':
    args = train_args()
    main(args)
