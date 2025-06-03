import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def clones_localattn(d_model, length, stride):
    '''
    Produce local attention projecting layers
    :param d_model: hidden dim
    :param length: input length
    :param stride: conv1d stride
    :return: layer list
    '''
    padding = 0 if int(math.ceil(length / stride) * stride) == length else int(math.ceil(length / stride) * stride) - length
    return nn.ModuleList([nn.Linear(d_model, d_model),
                          nn.Conv1d(d_model, d_model, kernel_size=stride, stride=stride, padding=padding),
                          nn.Conv1d(d_model, d_model, kernel_size=stride, stride=stride, padding=padding),
                          nn.Linear(d_model, d_model)])

class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn or local-attn, and feed forward (defined below)"
    def __init__(self, size, attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, selfattn=True, length=50, stride=5, dropout=0.1):
        """
        Take in model size and number of heads.
        :param h: head num
        :param d_model: hidden dim
        :param selfattn: True for self-attention, False for local attention
        :param length: input length
        :param stride: conv1d stride
        :param dropout:
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones_localattn(d_model, length, stride) if not selfattn else clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.selfattn = selfattn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if not self.selfattn:
            # calculate the q, k, v in local attention
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key.transpose(-2, -1), value.transpose(-2, -1)))]
            query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            key = key.transpose(-2, -1).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            value = value.transpose(-2, -1).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        else:
            # calculate the q, k, v in self-attention
            query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concat and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)