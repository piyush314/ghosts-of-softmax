"""Small transformer for fast stability experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    attentions: Optional[list[torch.Tensor]] = None


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dModel: int, nHeads: int, dropout: float = 0.0):
        super().__init__()
        self.nHeads = nHeads
        self.headDim = dModel // nHeads
        self.scale = self.headDim ** -0.5

        self.qkv = nn.Linear(dModel, 3 * dModel)
        self.proj = nn.Linear(dModel, dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, returnAttn: bool = False):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nHeads, self.headDim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scoresMasked = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scoresMasked, dim=-1)
        attnPreDrop = attn
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)

        if returnAttn:
            return out, attnPreDrop, scores  # Return pre-dropout weights and scores
        return out, None, None


class FeedForward(nn.Module):
    """FFN with configurable activation."""

    def __init__(self, dModel: int, dFF: int = None, dropout: float = 0.0):
        super().__init__()
        dFF = dFF or 4 * dModel
        self.fc1 = nn.Linear(dModel, dFF)
        self.fc2 = nn.Linear(dFF, dModel)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, dModel: int, nHeads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dModel)
        self.attn = Attention(dModel, nHeads, dropout)
        self.ln2 = nn.LayerNorm(dModel)
        self.ffn = FeedForward(dModel, dropout=dropout)

    def forward(self, x: torch.Tensor, returnAttn: bool = False):
        attnOut, attnWeights, attnScores = self.attn(self.ln1(x), returnAttn)
        x = x + attnOut
        x = x + self.ffn(self.ln2(x))
        return x, attnWeights, attnScores


class SmallTransformer(nn.Module):
    """Minimal transformer for stability experiments."""

    def __init__(
        self,
        vocabSize: int = 1000,
        dModel: int = 128,
        nHeads: int = 4,
        nLayers: int = 4,
        maxSeq: int = 128,
        dropout: float = 0.0,
        returnAttn: bool = False,
    ):
        super().__init__()
        self.dModel = dModel
        self.returnAttnDefault = returnAttn
        self.lastAttnWeights = None
        self.lastAttnScores = None

        self.wte = nn.Embedding(vocabSize, dModel)
        self.wpe = nn.Embedding(maxSeq, dModel)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(dModel, nHeads, dropout) for _ in range(nLayers)
        ])

        self.lnf = nn.LayerNorm(dModel)
        self.head = nn.Linear(dModel, vocabSize, bias=False)

        self.apply(self._initWeights)

    def _initWeights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        inputIds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        outputAttentions: bool = None,
        output_attentions: bool = None,
    ) -> TransformerOutput:
        if output_attentions is not None:
            outputAttentions = output_attentions
        if outputAttentions is None:
            outputAttentions = self.returnAttnDefault
        B, T = inputIds.shape
        pos = torch.arange(T, device=inputIds.device).unsqueeze(0)

        x = self.drop(self.wte(inputIds) + self.wpe(pos))

        attentions = [] if outputAttentions else None
        self.lastAttnWeights = None
        self.lastAttnScores = None

        for block in self.blocks:
            x, attn, scores = block(x, returnAttn=outputAttentions)
            if outputAttentions and attn is not None:
                attentions.append(attn)
                # Store last layer's attention for Var_α computation
                self.lastAttnWeights = attn
                self.lastAttnScores = scores

        x = self.lnf(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift = logits[:, :-1, :].contiguous()
            target = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift.view(-1, shift.size(-1)), target.view(-1))

        return TransformerOutput(logits=logits, loss=loss, attentions=attentions)
