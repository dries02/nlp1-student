import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbedAndPoolEncoder(nn.Module):
    """
    Embeds a sequence of token ids and pools over the sequence (mean, sum or max).
    """

    def __init__(self, embedding_matrix, pooling="mean", freeze_embeddings=True, padding_idx=0):
        super().__init__()
        if pooling not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.pooling = pooling
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx
        )

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    def forward(self, x):
        e = self.embedding(x)                          
        mask = x.ne(self.padding_idx).unsqueeze(-1)    
        if self.pooling == "max":
            return e.masked_fill(~mask, float("-inf")).max(dim=1).values
        summed = (e * mask).sum(dim=1)
        if self.pooling == "sum":
            return summed
        return summed / mask.sum(dim=1).clamp_min(1)


class RNNEncoder(nn.Module):
    """
    Reads the sequence word by word with an LSTM or GRU, so (unlike pooling) it sees word order.
    pooling="last": final hidden state (both directions if bidirectional)
    pooling="max" / "mean": max / mean over the hidden states of all words
    Padding is skipped with pack_padded_sequence so it doesn't influence the result.
    """

    def __init__(self, embedding_matrix, cell="lstm", hidden_size=128, bidirectional=True, num_layers=1,
                 dropout=0.0, pooling="max", freeze_embeddings=True, padding_idx=0):
        super().__init__()
        if cell not in ("lstm", "gru"):
            raise ValueError(f"Unknown cell: {cell}")
        if pooling not in ("last", "max", "mean"):
            raise ValueError(f"Unknown pooling: {pooling}")
        self.cell = cell
        self.pooling = pooling
        self.padding_idx = padding_idx
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=freeze_embeddings, padding_idx=padding_idx
        )
        rnn_class = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_class(
            input_size=embedding_matrix.shape[1], hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def enc_dim(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, x):
        lengths = x.ne(self.padding_idx).sum(dim=1).clamp_min(1)          
        e = self.embedding(x)                                              
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)

        if self.pooling == "last":
            h_n = h[0] if self.cell == "lstm" else h                       
            if self.bidirectional:
                return torch.cat([h_n[-2], h_n[-1]], dim=-1)              
            return h_n[-1]

        out, _ = pad_packed_sequence(out, batch_first=True)               
        mask = (torch.arange(out.shape[1], device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        if self.pooling == "max":
            return out.masked_fill(~mask, float("-inf")).max(dim=1).values
        return (out * mask).sum(dim=1) / lengths[:, None]


def make_mlp(in_dim, hidden_size, num_layers, dropout, num_classes=2):
    layers = []
    for _ in range(num_layers):
        layers += [nn.Dropout(dropout), nn.Linear(in_dim, hidden_size), nn.Tanh()]
        in_dim = hidden_size
    layers += [nn.Dropout(dropout), nn.Linear(in_dim, num_classes)]
    return nn.Sequential(*layers)


class PairClassifier(nn.Module):
    """
    Siamese encodes both sentences separately with the same encoder (u, v), combines them and classifies with an MLP.
    combine="concat" is [u; v]
    combine="full" is   [u; v; |u - v|; u * v]  
    """

    def __init__(self, encoder, combine="full", hidden_size=128, num_layers=1, dropout=0.0, num_classes=2):
        super().__init__()
        if combine not in ("concat", "full"):
            raise ValueError(f"Unknown combine: {combine}")
        self.encoder = encoder
        self.combine = combine
        in_dim = encoder.enc_dim * (2 if combine == "concat" else 4)
        self.mlp = make_mlp(in_dim, hidden_size, num_layers, dropout, num_classes)

    def combine_features(self, u, v):
        if self.combine == "concat":
            return torch.cat([u, v], dim=-1)
        return torch.cat([u, v, (u - v).abs(), u * v], dim=-1)

    def forward(self, x1, x2):
        u = self.encoder(x1)
        v = self.encoder(x2)
        return self.mlp(self.combine_features(u, v))  

class JointClassifier(nn.Module):
    """
    Joint: one encoder reads sentence1 [SEP] sentence2 as a single sequence so while reading
    sentence2 it already knows sentence1. x1 must contain the joined sequence; x2 is ignored.
    """

    def __init__(self, encoder, hidden_size=128, num_layers=1, dropout=0.0, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.mlp = make_mlp(encoder.enc_dim, hidden_size, num_layers, dropout, num_classes)

    def forward(self, x1, x2=None):
        return self.mlp(self.encoder(x1))
