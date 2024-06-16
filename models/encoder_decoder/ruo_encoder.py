import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn.functional as F


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, head,d_model,input_dim , dropout,depth):
        super(Encoder, self).__init__()
        self.head = int(head)
        self.d_model = int(d_model),
        self.dropout = dropout,
        self.input_dim =int(input_dim),
        self.depth = int(depth)
        self.layers = nn.ModuleList([
            EncoderLayer(head = self.head,d_model = d_model,input_dim =input_dim , dropout=dropout
            ) for _ in range(self.depth)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."

        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self,head,d_model,input_dim , dropout):
        super(EncoderLayer, self).__init__()
        self.feed_forward = PositionwiseFeedForward(d_model,input_dim,dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout),SublayerConnection(d_model, dropout)])
        self.self_attn = MultiHeadedAttention(head, d_model, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)

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

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self,  num_heads,embed_dim, dropout_prob=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.head_dim = embed_dim // num_heads

        # Linear transformations for queries, keys, and values for each head
        self.query_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.key_linear = torch.nn.Linear(embed_dim, embed_dim)
        self.value_linear = torch.nn.Linear(embed_dim, embed_dim)

        # Output linear transformation
        self.out_linear = torch.nn.Linear(embed_dim, embed_dim)

        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout_prob)

    def split_heads(self, x):
        try:
            batch_size, patch_size, embed_dim = x.size()
        except:
            batch_size = 1
            patch_size, embed_dim = x.size()
        return x.view(batch_size, patch_size, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x):
        try:
            batch_size, patch_size, embed_dim = x.size()
        except:
            batch_size=1
            patch_size, embed_dim = x.size()


        # Linearly transform queries, keys, and values for each head
        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)

        # Split heads
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        # Calculate scaled dot-product attention scores
        attention_scores = torch.matmul(queries, keys.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        # Apply dropout to attention scores
        attention_scores = self.dropout(attention_scores)

        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum using attention weights
        attention_output = torch.matmul(attention_weights, values)

        # Merge heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, patch_size, embed_dim)

        # Linearly transform the merged output
        output = self.out_linear(attention_output)

        return output
