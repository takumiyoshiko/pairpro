from typing import Dict

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 word_to_id: Dict[str, int],
                 embedding:torch.Tensor,
                 n_in:int=256,
                 n_mid_units:int=128,
                 n_out:int=2
                 ) -> None:
        # お約束
        super(MLP, self).__init__()
        self.n_in = n_in
        # パラメータを持つ層を登録
        # Linear は W x + b の線形変換 (全結合) を行う
        # 第1引数は入力の次元数
        # 第2引数は出力の次元数
        self.l1 = nn.Linear(n_in, n_mid_units)
        self.l2 = nn.Linear(n_mid_units, n_out)
        self.tanh = nn.Tanh()
        self.embd = nn.Embedding(len(word_to_id),n_in)
        self._set_initial_embeddings(embedding)

    def forward(self, x, mask):
        # データを受け取った際のforward計算を書く
        # relu は Rectified Linear Unit: f(x)=max(0,x)
        e  = self.embd(x)
        x2 = e.sum(dim=1)
        mask2 = mask.sum(dim=1, keepdim=True, dtype=torch.float32)
        x3 = x2/mask2
        h1 = self.tanh(self.l1(x3))
        h2 = self.l2(h1)
        return h2

    def _set_initial_embeddings(self, weights):
        self.embd.weight = nn.Parameter(weights, requires_grad=False)
