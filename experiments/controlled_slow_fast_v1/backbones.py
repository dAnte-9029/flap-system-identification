"""History encoding and one-token causal updates for the matched 20 ms baselines."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LSTMBackbone(nn.Module):
    width = 45

    def __init__(self, history_dim, context_dim):
        super().__init__()
        self.encoder = nn.LSTM(history_dim, 52, batch_first=True)
        self.hidden = nn.Linear(52, self.width)
        self.memory = nn.Linear(52, self.width)
        self.cell = nn.LSTMCell(context_dim, self.width)

    def initialize(self, history, steps):
        _, (h, c) = self.encoder(history)
        return torch.tanh(self.hidden(h[0])), self.memory(c[0])

    def step(self, context, memory):
        h, c = self.cell(context, memory)
        return h, (h, c)


class DilatedBlock(nn.Module):
    """Residual causal convolution with three taps at [-2d, -d, 0]."""
    def __init__(self, width, dilation):
        super().__init__()
        self.dilation = dilation
        self.filter = nn.Linear(3 * width, width)
        self.norm = nn.LayerNorm(width)

    def combine(self, old, middle, current):
        taps = torch.cat([old, middle, current], -1)
        return self.norm(current + F.silu(self.filter(taps)))

    def initialize(self, x):
        d, length = self.dilation, x.size(1)
        padded = F.pad(x, (0, 0, 2*d, 0))
        y = self.combine(padded[:, :length], padded[:, d:d+length], x)
        return y, list(padded[:, -2*d:].unbind(1))

    def step(self, x, cache):
        y = self.combine(cache[0], cache[self.dilation], x)
        return y, cache[1:] + [x]


class TCNBackbone(nn.Module):
    width = 51

    def __init__(self, history_dim, context_dim):
        super().__init__()
        self.history_input = nn.Linear(history_dim, self.width)
        self.step_input = nn.Linear(context_dim, self.width)
        self.layers = nn.ModuleList(DilatedBlock(self.width, d) for d in (1, 2, 4, 8))

    def initialize(self, history, steps):
        x = self.history_input(history)
        caches = []
        for layer in self.layers:
            x, cache = layer.initialize(x)
            caches.append(cache)
        return caches

    def step(self, context, memory):
        x = self.step_input(context)
        caches = []
        for layer, cache in zip(self.layers, memory):
            x, new_cache = layer.step(x, cache)
            caches.append(new_cache)
        return x, caches


class AttentionBlock(nn.Module):
    """Pre-norm causal decoder block, with rolling key/value state for rollout."""
    def __init__(self, width, heads, window):
        super().__init__()
        self.heads, self.window = heads, window
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 68), nn.GELU(), nn.Linear(68, width))

    def forward(self, x, past=None):
        batch, length, width = x.shape
        q, k, v = (t.reshape(batch, length, self.heads, width//self.heads).transpose(1, 2)
                   for t in self.qkv(self.norm1(x)).chunk(3, -1))
        mask = None
        causal = past is None
        if past is not None:
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
        elif length > self.window:
            i = torch.arange(length, device=x.device)
            lag = i[:, None] - i[None, :]
            mask, causal = (lag >= 0) & (lag < self.window), False
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        x = x + self.output(attn.transpose(1, 2).reshape(batch, length, width))
        x = x + self.ff(self.norm2(x))
        return x, (k[:, :, -(self.window-1):], v[:, :, -(self.window-1):])


class TransformerBackbone(nn.Module):
    width = 48

    def __init__(self, history_dim, context_dim):
        super().__init__()
        self.history_input = nn.Linear(history_dim, self.width)
        self.step_input = nn.Linear(context_dim, self.width)
        self.layers = nn.ModuleList(AttentionBlock(self.width, 3, 32) for _ in range(2))
        self.final_norm = nn.LayerNorm(self.width)
        self.register_buffer('frequencies', torch.exp(-math.log(10000.)*torch.arange(0,self.width,2)/self.width))

    def initialize(self, history, steps):
        length = history.size(1)
        angle = torch.arange(length + steps, device=history.device)[:, None] * self.frequencies
        pos = torch.stack([angle.sin(), angle.cos()], -1).flatten(-2)
        x = self.history_input(history) + pos[:length]
        caches = []
        for layer in self.layers:
            x, cache = layer(x)
            caches.append(cache)
        return caches, pos, length

    def step(self, context, memory):
        caches, pos, index = memory
        x = (self.step_input(context) + pos[index]).unsqueeze(1)
        updated = []
        for layer, cache in zip(self.layers, caches):
            x, cache = layer(x, cache)
            updated.append(cache)
        return self.final_norm(x[:, 0]), (updated, pos, index+1)


BACKBONES = {'lstm': LSTMBackbone, 'tcn': TCNBackbone, 'transformer': TransformerBackbone}
