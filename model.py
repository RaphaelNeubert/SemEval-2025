import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DecoderLayer(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims, dropout=0):
        super().__init__()
        self.mha1 = MultiHeadAttention(dim_embeddings, num_heads, dropout=dropout)
        self.mha2 = MultiHeadAttention(dim_embeddings, num_heads, dropout=dropout)
        self.ffn = FeedForwardNetwork(dim_embeddings, hidden_dims, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_embeddings)
        self.norm2 = nn.LayerNorm(dim_embeddings)
        self.norm3 = nn.LayerNorm(dim_embeddings)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x_norm = self.norm1(x)
        x = x + self.mha1(x_norm, x_norm, x_norm, mask=tgt_mask, causal=True)
        enc_norm = self.norm2(enc_output)
        x = x + self.mha2(self.norm2(x), enc_norm, enc_norm, mask=src_mask)
        x = x + self.ffn(self.norm3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim_embeddings, num_heads, hidden_dims, num_layers, dropout=0):
        super().__init__()
        self.dec_layers = nn.ModuleList([DecoderLayer(dim_embeddings, num_heads, hidden_dims, dropout=dropout)
                                         for _ in range(num_layers)])
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.dec_layers:
            out = layer(tgt, enc_output, src_mask, tgt_mask)
        return out
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, dim_embeddings, num_heads, ffn_hidden_dims, 
                 num_encoder_layers, num_decoder_layers, max_seq_len=5000, dropout=0):
        super().__init__()
        self.dim_embeddings = dim_embeddings
        self.tgt_vocab_size = tgt_vocab_size
        self.src_embedding = nn.Embedding(src_vocab_size, dim_embeddings)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dim_embeddings)
        self.positional_encoding = PositionalEncoding(dim_embeddings, max_seq_len, dropout)
        self.encoder = Encoder(dim_embeddings, num_heads, ffn_hidden_dims, num_encoder_layers, dropout)
        self.decoder = Decoder(dim_embeddings, num_heads, ffn_hidden_dims, num_decoder_layers, dropout)
        self.fc = nn.Linear(dim_embeddings, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        enc_out = self.encoder(src, mask=src_mask)

        tgt_seq_len = tgt.shape[1]
        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        out = self.fc(out.view(-1, self.dim_embeddings))
        out = out.view(-1, tgt_seq_len, self.tgt_vocab_size)
        #out = F.softmax(out, dim=-1)
        return out

    def inference(self, src, start_index, end_index, src_mask=None, max_len=1000, temperature=0):
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        enc_out = self.encoder(src, mask=src_mask)

        tgt_indices = torch.tensor([[start_index]], dtype=torch.long, device=src.device)

        for _ in range(max_len-1):
            tgt = self.tgt_embedding(tgt_indices)
            tgt = self.positional_encoding(tgt)

            out = self.decoder(tgt, enc_out, src_mask=src_mask)
            out = self.fc(out[:, -1, :])
            if temperature == 0:
                next_token = torch.argmax(out, dim=-1).item()
            else:
                probabilities = torch.softmax(out/temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).item()

            tgt_indices = torch.cat((tgt_indices, torch.tensor([[next_token]], device=src.device)), dim=1)
            if next_token == end_index:
                break

        return tgt_indices



class SemModel(nn.Module):
    def __init__(self, vocab_size, num_classes, dim_embeddings, num_heads, ffn_hidden_dims, 
                 num_encoder_layers, max_seq_len=5000, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim_embeddings)
        self.positional_encoding = PositionalEncoding(dim_embeddings, max_seq_len, dropout)
        self.encoder = Encoder(dim_embeddings, num_heads, ffn_hidden_dims, num_encoder_layers, dropout)
        self.fc = nn.Linear(dim_embeddings, num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        enc_out = self.encoder(x, mask=mask)
        out = self.fc(enc_out[:,0,:])
        return out

