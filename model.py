import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_classes: int = 5
    dim_embeddings: int = 512
    num_heads: int = 8
    feed_forward_hidden_dims: int = 512*4
    num_encoder_layers: int = 6
    dropout: int = 0.1
    max_seq_len: int = 5000

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embeddings, max_len=5000, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len)
        # apply log rule to original formula for computational speedup
        div = torch.exp(-torch.arange(0, dim_embeddings, 2) / dim_embeddings * torch.log(torch.tensor(10000)))

        pos_enc = torch.zeros(max_len, dim_embeddings)
        pos_enc[:, ::2] = torch.sin(pos.unsqueeze(-1)/div)
        pos_enc[:, 1::2] = torch.cos(pos.unsqueeze(-1)/div)
        # move automatically to specific device when module is moved
        self.register_buffer("pos_enc", pos_enc)

    # expected shape: (batch_size, sequence_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        x = x + self.pos_enc[:x.shape[1],:].expand(x.shape[0], -1, -1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embeddings, num_heads, dropout=0):
        super().__init__()
        assert dim_embeddings % num_heads == 0

        self.Q_linear = nn.Linear(dim_embeddings, dim_embeddings)
        self.K_linear = nn.Linear(dim_embeddings, dim_embeddings)
        self.V_linear = nn.Linear(dim_embeddings, dim_embeddings)
        self.O_linear = nn.Linear(dim_embeddings, dim_embeddings)
        
        self.num_heads = num_heads
        self.dim_head = dim_embeddings // num_heads
        # avoid small gradients in softmax caused by large dot products when calculating the scores
        self.scale = 1 / (self.dim_head ** 0.5) 

        self.dropout = nn.Dropout(dropout)

    # expected shape of Q, K, V: (batch_size, seq_len, dim_embeddings)
    # expected shape of mask: (batch_size, seq_len_key)
    def forward(self, Q, K, V, mask=None, causal=False):
        in_shape = Q.shape
        Q = self.Q_linear(Q)
        K = self.K_linear(K)
        V = self.V_linear(V)
        # transform shape to (batch_size, num_heads, seq_len, dim_head)
        Q = Q.view(Q.shape[0], Q.shape[1], self.num_heads, self.dim_head).transpose(1,2)
        K = K.view(K.shape[0], K.shape[1], self.num_heads, self.dim_head).transpose(1,2)
        V = V.view(V.shape[0], V.shape[1], self.num_heads, self.dim_head).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-1,-2)) * self.scale
        # scores.shape: (batch_size, num_heads, seq_len_query, seq_len_key)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        if causal: # mask away information of future sequence elements (required for training)
            causal_mask = torch.triu(torch.full_like(scores, float("-inf"), device=Q.device), diagonal=1)
            scores += causal_mask

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1,2).contiguous().view(in_shape)
        out = self.O_linear(out)
        return self.dropout(out)


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_embeddings, hidden_dims, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(dim_embeddings, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, dim_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims, dropout=0):
        super().__init__()
        self.mha = MultiHeadAttention(dim_embeddings, num_heads, dropout=dropout)
        self.ffn = FeedForwardNetwork(dim_embeddings, hidden_dims, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_embeddings)
        self.norm2 = nn.LayerNorm(dim_embeddings)
        
    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        x = x + self.mha(x_norm, x_norm, x_norm, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims, num_layers, dropout=0):
        super().__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(dim_embeddings, num_heads, hidden_dims, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x


class SemEvalModel(nn.Module):
    def __init__(self, vocab_size, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.dim_embeddings)
        self.positional_encoding = PositionalEncoding(config.dim_embeddings, config.max_seq_len, config.dropout)
        self.encoder = Encoder(config.dim_embeddings, config.num_heads, 
                               config.feed_forward_hidden_dims, config.num_encoder_layers, config.dropout)
        self.fc = nn.Linear(config.dim_embeddings, config.num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        enc_out = self.encoder(x, mask=mask)
        out = self.fc(enc_out[:,0,:])
        return out

