import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizerFast
from utils.device_utils import get_device

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
sep_id = tokenizer.sep_token_id  # → 102


class JointEmbedding(nn.Module):
    def __init__(self, vocab_size, size):
        super(JointEmbedding, self).__init__()
        self.size = size
        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding(2, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, input_tensor):
        pos_tensor = self.attention_position(self.size, input_tensor)

        sep_pos = (input_tensor == sep_id).int().argmax(dim=1)
        sep_pos[sep_pos == 0] = input_tensor.size(1)

        arange = torch.arange(
            input_tensor.size(1), device=input_tensor.device
        ).unsqueeze(0)
        segment_tensor = (arange > sep_pos.unsqueeze(1)).long()

        output = (
            self.token_emb(input_tensor) + self.segment_emb(segment_tensor) + pos_tensor
        )
        return self.norm(output)

    def attention_position(self, dim, input_tensor):
        batch_size = input_tensor.size(0)
        sentence_size = input_tensor.size(-1)
        pos = torch.arange(sentence_size, dtype=torch.long, device=input_tensor.device)
        d = torch.arange(dim, dtype=torch.long, device=input_tensor.device)
        d = 2 * d / dim
        pos = pos.unsqueeze(1)
        pos = pos / (1e4**d)
        pos[:, ::2] = torch.sin(pos[:, ::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        return pos.expand(batch_size, *pos.size())


class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()
        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor = None):
        query, key, value = (
            self.q(input_tensor),
            self.k(input_tensor),
            self.v(input_tensor),
        )
        scale = query.size(-1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale  # [B, L, L]

        if attention_mask is not None:
            # Ensure bool dtype and proper broadcast shape
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            # Expecting [B, 1, 1, L] or [B, 1, L, L]; collapse extra dims
            if attention_mask.dim() > 3:
                attention_mask = attention_mask.squeeze(1)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # [B,1,L,L]
            # Mask out pads (False = pad)
            scores = scores.masked_fill(~attention_mask.squeeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)  # [B, L, D]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_inp, dim_out) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(dim_out * num_heads, dim_inp)
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        attn_mask = self._prepare_attention_mask(attention_mask, input_tensor)
        s = [head(input_tensor, attn_mask) for head in self.heads]
        scores = torch.cat(s, dim=-1)
        scores = self.linear(scores)
        return self.norm(scores)

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: torch.Tensor, input_tensor: torch.Tensor
    ):
        if attention_mask is None:
            return None
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        if attention_mask.dim() == 2:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        elif attention_mask.dim() == 3:
            mask = attention_mask.unsqueeze(1)  # [B,1,L,L]
        else:
            raise ValueError("Attention mask must have 2 or 3 dimensions.")
        return mask.to(device=input_tensor.device, dtype=torch.bool)


class Encoder(nn.Module):
    def __init__(self, dim_inp, dim_out, attention_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(attention_heads, dim_inp, dim_out)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_inp, dim_out),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_out, dim_inp),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        context = self.attention(input_tensor, attention_mask)
        res = self.feed_forward(context)
        return self.norm(res)


class BERT(nn.Module):
    def __init__(self, vocab_size, dim_inp, dim_out, attention_heads=4, device=None):
        super(BERT, self).__init__()
        self.device = device or get_device()
        self.embedding = JointEmbedding(vocab_size, dim_inp)
        self.encoder = Encoder(dim_inp, dim_out, attention_heads)
        self.token_prediction_layer = nn.Linear(dim_inp, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.classification_layer = nn.Linear(dim_inp, 2)
        self.to(self.device)

    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.embedding(input_tensor)
        encoded = self.encoder(embedded, attention_mask)
        token_predictions = self.token_prediction_layer(encoded)
        first_word = encoded[:, 0, :]
        return self.softmax(token_predictions), self.classification_layer(first_word)
