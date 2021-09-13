import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from src.pgn_cn.utils import masked_softmax
from typing import Optional


class RNNEncoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           bidirectional=True)
        self.h_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.c_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        hidden_state, (hn, cn) = self.rnn(x)
        hidden_state, _ = pad_packed_sequence(hidden_state,
                                              batch_first=True,
                                              total_length=lengths.max().item())
        last_hidden_cat = torch.cat([hn[0], hn[1]], dim=-1)
        last_cell_cat = torch.cat([cn[0], cn[1]], dim=-1)
        init_dec_hidden = self.h_proj(last_hidden_cat)
        init_dec_cell = self.c_proj(last_cell_cat)
        return hidden_state, init_dec_hidden, init_dec_cell


class Attention(nn.Module):

    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.proj_enc_h = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.proj_dec_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_coverage_vec = nn.Linear(1, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, enc_h: torch.Tensor, dec_h: torch.Tensor,
                coverage_vec: torch.Tensor, mask: torch.Tensor):
        proj_sum = self.proj_enc_h(enc_h) + self.proj_dec_h(dec_h).unsqueeze(dim=1).expand(-1, enc_h.size(1), -1)
        if coverage_vec is not None: proj_sum += self.proj_coverage_vec(coverage_vec.unsqueeze(dim=-1))
        e = self.proj(torch.tanh(proj_sum + self.bias)).squeeze(dim=-1)  # (B, T)
        a = masked_softmax(e, mask)
        a_comb_cov = torch.cat([a.unsqueeze(dim=2), coverage_vec.unsqueeze(dim=2)], dim=-1) \
            if coverage_vec is not None else None
        covloss = a_comb_cov.min(dim=-1)[0].sum(dim=-1) if coverage_vec is not None else None
        h_star = torch.bmm(a.unsqueeze(dim=1), enc_h).squeeze(dim=1)
        new_coverage_vec = coverage_vec + a if coverage_vec is not None else None
        return a, h_star, new_coverage_vec, covloss


class DecodeOneStep(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, vocab_size: int):
        super(DecodeOneStep, self).__init__()
        self.rnn = nn.LSTMCell(input_size=input_size,
                               hidden_size=hidden_size)
        self.attn = Attention(hidden_size)
        self.vocab_fc = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size),
            nn.Linear(2 * hidden_size, vocab_size)
        )
        self.proj_pgen = nn.Linear(3 * hidden_size + input_size, 1)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, cn: torch.Tensor, enc_h: torch.Tensor,
                coverage_vec: torch.Tensor, mask: torch.Tensor, x_extend: Optional[torch.Tensor],
                p_extend: torch.Tensor, y: Optional[torch.Tensor] = None, is_training: bool = True):
        hn, cn = self.rnn(x, (hn, cn))
        a, h_star, coverage_vector, covloss = self.attn(enc_h, hn, coverage_vec, mask)
        p_gen = torch.sigmoid(self.proj_pgen(torch.cat([h_star, hn, x], dim=-1)))
        logits = self.vocab_fc(torch.cat([h_star, hn], dim=-1))
        p = masked_softmax(logits) * p_gen
        p = torch.cat([p, p_extend], dim=-1)
        p.scatter_add_(1, x_extend, a * (1 - p_gen))
        if not is_training: return p, hn, cn, coverage_vector, covloss
        p_star = p.gather(1, y.view(-1, 1)).squeeze()
        return p_star, hn, cn, coverage_vector, covloss


class Decoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 vocab_size: int):
        super(Decoder, self).__init__()
        self.step = DecodeOneStep(input_size=input_size, hidden_size=hidden_size,
                                  vocab_size=vocab_size)

    def forward(self, article_extend_vocab: torch.Tensor, dec_inp: torch.Tensor, dec_tgt: torch.Tensor,
                enc_h: torch.Tensor, hn: torch.Tensor, cn: torch.Tensor, p_extend: torch.Tensor,
                mask: torch.Tensor, lambda_: float, is_coverage: bool):
        coverage_vector = torch.zeros_like(mask, dtype=torch.float) if is_coverage else None
        Ps = []
        if is_coverage: Covloss = []
        for x, y in zip(dec_inp, dec_tgt):
            p, hn, cn, coverage_vector, covloss = self.step(x, hn, cn, enc_h, coverage_vector,
                                                            mask, article_extend_vocab, p_extend, y)
            Ps.append(p)
            if is_coverage: Covloss.append(covloss)

        label_mask = dec_tgt != 0  # (T, B)
        loss = -(torch.log(torch.stack(Ps)+1e-30) * label_mask).sum(dim=0) / label_mask.sum(dim=0)
        if is_coverage:
            coverage_loss = (lambda_ * torch.stack(Covloss) * label_mask).sum(dim=0) / label_mask.sum(dim=0)
            total_loss = loss + coverage_loss
            return (total_loss, loss, coverage_loss)
        return (loss,)


class Summary(nn.Module):

    def __init__(self, weight: torch.Tensor, embedding_dim: int,
                 hidden_size: int, vocab_size: int):
        super(Summary, self).__init__()
        self.emb = nn.Embedding.from_pretrained(weight)
        self.encoder = RNNEncoder(input_size=embedding_dim, hidden_size=hidden_size)
        self.decoder = Decoder(input_size=embedding_dim, hidden_size=hidden_size,
                               vocab_size=vocab_size)

    def forward(self, article: torch.Tensor, article_extend_vocab: torch.Tensor,
                summary_inp: torch.Tensor, summary_tgt: torch.Tensor,
                lambda_: float, is_coverage: bool, max_article_oovs: int):
        mask = article == 0
        bs = article.size(0)
        p_extend = torch.zeros((bs, max_article_oovs), dtype=torch.float).to(article.device)
        lengths = (~mask).sum(dim=-1)
        # 假如 bs 为 64 ，共有 4 块 GPU 那么一次性经过 collate 函数 256 个样本，
        # 因此也只根据这 256 个里最大的样本进行了截断，导致分到每个 GPU 里仍然存在
        # 最长的样本仍有 pad
        max_length = lengths.max().item()
        article = article[:, :max_length]
        article_extend_vocab = article_extend_vocab[:, :max_length]
        mask = article == 0
        summary_inp.t_()
        summary_tgt.t_()
        article_emb = self.emb(article)
        dec_inp_emb = self.emb(summary_inp)
        enc_h, dec_h, dec_c = self.encoder(article_emb, lengths)
        loss = self.decoder(article_extend_vocab, dec_inp_emb, summary_tgt,
                            enc_h, dec_h, dec_c, p_extend, mask, lambda_, is_coverage)
        return loss
