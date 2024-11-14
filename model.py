import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, dim_embeddings, max_len=5000):
        super().__init__()
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
        return x + self.pos_enc[:x.shape[1],:].expand(x.shape[0], -1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embeddings, num_heads):
        super().__init__()
        assert dim_embeddings % num_heads == 0

        self.Q_linear = nn.Linear(dim_embeddings, dim_embeddings)
        self.K_linear = nn.Linear(dim_embeddings, dim_embeddings)
        self.V_linear = nn.Linear(dim_embeddings, dim_embeddings)
        self.O_linear = nn.Linear(dim_embeddings, dim_embeddings)
        
        self.num_heads = num_heads
        self.dim_head = dim_embeddings // num_heads
        # avoid small gradients in softmax caused by large dot products when calculating the scores
        self.scale = 1 / (dim_embeddings ** 0.5) 

    # expected shape of Q, K, V: (batch_size, seq_len, dim_embeddings)
    # expected shape of mask: (batch_size, seq_len)
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

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        if causal: # mask away information of future sequence elements (required for training)
            causal_mask = torch.triu(torch.full_like(scores, float("-inf"), device=Q.device), diagonal=1)
            scores += causal_mask

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1,2).contiguous().view(in_shape)
        out = self.O_linear(out)
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, dim_embeddings, hidden_dims):
        super().__init__()
        self.fc1 = nn.Linear(dim_embeddings, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, dim_embeddings)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims):
        super().__init__()
        self.mha = MultiHeadAttention(dim_embeddings, num_heads)
        self.ffn = FeedForwardNetwork(dim_embeddings, hidden_dims)
        self.norm1 = nn.LayerNorm(dim_embeddings)
        self.norm2 = nn.LayerNorm(dim_embeddings)
        
    def forward(self, x, mask=None):
        x_norm = self.norm1(x)
        x = x + self.mha(x_norm, x_norm, x_norm, mask)
        x = x + self.ffn(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims, num_layers):
        super().__init__()
        self.enc_layers = nn.ModuleList([EncoderLayer(dim_embeddings, num_heads, hidden_dims)
                                         for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.enc_layers:
            x = layer(x, mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims):
        super().__init__()
        self.mha1 = MultiHeadAttention(dim_embeddings, num_heads)
        self.mha2 = MultiHeadAttention(dim_embeddings, num_heads)
        self.ffn = FeedForwardNetwork(dim_embeddings, hidden_dims)
        self.norm1 = nn.LayerNorm(dim_embeddings)
        self.norm2 = nn.LayerNorm(dim_embeddings)
        self.norm3 = nn.LayerNorm(dim_embeddings)
        
    def forward(self, x, enc_output):
        x_norm = self.norm1(x)
        x = x + self.mha1(x_norm, x_norm, x_norm, causal=True)
        enc_norm = self.norm2(enc_output)
        x = x + self.mha2(self.norm2(x), enc_norm, enc_norm)
        x = x + self.ffn(self.norm3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims, num_layers):
        super().__init__()
        self.dec_layers = nn.ModuleList([DecoderLayer(dim_embeddings, num_heads, hidden_dims)
                                         for _ in range(num_layers)])
    def forward(self, x, enc_output):
        for layer in self.dec_layers:
            x = layer(x, enc_output)
        return x

d_emb = 100 
heads = 5
hidden_dim = d_emb*4
num_layers = 4
enc = Encoder(d_emb, heads, hidden_dim, num_layers).to("cuda")
dec = Decoder(d_emb, heads, hidden_dim, num_layers).to("cuda")
x = torch.randn(64, 100, d_emb, device="cuda")
enc_out = enc(x)
dec_out = dec(x, enc_out)
print(enc_out.shape)
print(dec_out.shape)
