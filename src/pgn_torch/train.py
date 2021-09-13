import copy
import os

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pgn_torch.dataset import PairDataset, SampleDataset, collate_fn
from src.pgn_torch.layers import PGN
from src.pgn_torch.utils import ProgressBar, ScheduledSampler
from src.utils.config import vocab_path
from src.utils.params_utils import get_params
from src.utils.wv_loader import Vocab


def train(train_iter, model, v, teacher_forcing, params):
    """
    Args:
        dataset (dataset.PairDataset)
        val_dataset (dataset.PairDataset)
        v (vocab.Vocab)
        start_epoch (int, optional)
    """
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    pbar_batch = ProgressBar(n_total=len(train_iter), desc="Traing_batch")
    batch_loss = 0
    batch_losses = []
    for batch, data in enumerate(train_iter):
        x, y, x_len, y_len, oov, len_oovs= data
        if params['is_cuda']:
            x = x.to(params['device'])
            y = y.to(params['device'])

            len_oovs = len_oovs.to(params['device'])
        loss = model(x,
                     y,
                     len_oovs,
                     teacher_forcing=teacher_forcing)
        batch_losses.append(loss.item())
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=params['max_grad_norm'])
        optimizer.step()
        optimizer.zero_grad()
        batch_loss += loss.item()
        print('loss:', loss.item())
        pbar_batch(batch, {"batch_loss": batch_loss / (batch + 1)})

    batch_losses = np.mean(batch_losses)
    return batch_losses


def evaluate(model, eval_iter, params):
    """
    Args:
        model (torch.nn.Module)
        val_data (dataset.PairDataset)
    """
    val_loss = []
    model.eval()
    with torch.no_grad():
        device = params['device']
        for batch, data in enumerate(tqdm(eval_iter)):
            x, y, x_len, y_len, oov, len_oovs = data
            if params['is_cuda']:
                x = x.to(device)
                y = y.to(device)
                x_len = x_len.to(device)
                len_oovs = len_oovs.to(device)
            loss = model(x,
                         y,
                         len_oovs,
                         teacher_forcing=False)
            val_loss.append(loss.item())
    return np.mean(val_loss)


def main(params):
    DEVICE = torch.device('cuda') if params['is_cuda'] else torch.device('cpu')
    vocab = Vocab(vocab_path)
    train_data = SampleDataset(params, 'train', vocab, PairDataset)
    train_iter = DataLoader(dataset=train_data,
                            batch_size=params['batch_size'],
                            shuffle=True,
                            collate_fn=collate_fn)

    test_data = SampleDataset(params, 'val', vocab, PairDataset)
    test_iter = DataLoader(dataset=test_data,
                           batch_size=params['batch_size'],
                           shuffle=True,
                           collate_fn=collate_fn)
    model = PGN(vocab)
    model.to(DEVICE)

    #
    best_val_loss = np.inf
    start_epoch = 0
    num_epochs = len(range(start_epoch, params['epochs']))
    scheduled_sampler = ScheduledSampler(num_epochs)
    if params['scheduled_sampling']:
        print('动态 Teather forcing 模式打开')
    pbar_epoch = ProgressBar(n_total=len(train_iter), desc="Traing_epoch")
    for epoch in range(start_epoch, params['epochs']):
        model.train()
        # Teacher Forcing模式
        if params['scheduled_sampling']:
            teacher_forcing = scheduled_sampler.teacher_forcing(epoch - start_epoch)
        else:
            teacher_forcing = True
        print('teacher_forcing = {}'.format(teacher_forcing))
        # 训练
        batch_loss = train(train_iter, model, vocab, teacher_forcing, params)
        pbar_epoch(epoch, {"epoch_loss": batch_loss})
        # 验证
        val_loss = evaluate(model, test_iter, params)
        print('validation loss:{}'.format(val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.module if hasattr(model, "module") else model)
            torch.save(best_model.state_dict(), os.path.join(params['checkpoint_dir'], "best_model.pkl"))


if __name__ == "__main__":
    params = get_params()
    main(params)
