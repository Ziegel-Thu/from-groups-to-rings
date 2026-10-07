"""无 LayerNorm 的元素级 Transformer。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """标准多头自注意力。"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须被 n_heads 整除")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden):
        batch, length, width = hidden.shape
        query = self.w_q(hidden).view(
            batch, length, self.n_heads, self.d_head
        ).transpose(1, 2)
        key = self.w_k(hidden).view(
            batch, length, self.n_heads, self.d_head
        ).transpose(1, 2)
        value = self.w_v(hidden).view(
            batch, length, self.n_heads, self.d_head
        ).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_head)
        weights = F.softmax(scores, dim=-1)
        output = (weights @ value).transpose(1, 2).contiguous()
        return self.w_o(output.view(batch, length, width))


class MLP(nn.Module):
    """两层 GELU MLP。"""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.w_in = nn.Linear(d_model, d_mlp, bias=False)
        self.w_out = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, hidden):
        return self.w_out(F.gelu(self.w_in(hidden)))


class TransformerBlock(nn.Module):
    """无 LayerNorm 的残差 Transformer block。"""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, hidden):
        hidden = hidden + self.attn(hidden)
        return hidden + self.mlp(hidden)


class ElementTransformer(nn.Module):
    """输入 [a, SEP, b] 并预测代数乘积。"""

    def __init__(
        self,
        ring_size: int,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.ring_size = ring_size
        self.token_embedding = nn.Embedding(ring_size + 1, d_model)
        self.position_embedding = nn.Embedding(3, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(d_model, ring_size, bias=False)

    @staticmethod
    def _replace_position(hidden, state, position: int):
        output = hidden.clone()
        output[:, position, :] = state
        return output

    def forward(
        self,
        tokens,
        return_cache: bool = False,
        patch_site: str | None = None,
        patch_state=None,
        patch_position: int = 2,
    ):
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :]
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        cache = {"resid_pre": hidden}
        if patch_site == "resid_pre":
            hidden = self._replace_position(hidden, patch_state, patch_position)

        for layer, block in enumerate(self.blocks):
            hidden = hidden + block.attn(hidden)
            if layer == 0:
                cache["resid_mid"] = hidden
                if patch_site == "resid_mid":
                    hidden = self._replace_position(
                        hidden, patch_state, patch_position
                    )
            hidden = hidden + block.mlp(hidden)
            if layer == 0:
                cache["resid_post"] = hidden
                if patch_site == "resid_post":
                    hidden = self._replace_position(
                        hidden, patch_state, patch_position
                    )

        logits = self.unembed(hidden[:, 2, :])
        return (logits, cache) if return_cache else logits

