import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PairDataset(Dataset):
    """Holds the token ids of sentence1 and sentence2 and the label for each pair."""

    def __init__(self, ids1, ids2, labels):
        self.x1 = [torch.tensor(x, dtype=torch.long) for x in ids1]
        self.x2 = [torch.tensor(x, dtype=torch.long) for x in ids2]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x1[i], self.x2[i], self.y[i]


def collate_pairs(batch, padding_value=0):
    """Pads sentence1 and sentence2 separately to the longest sentence in the batch."""
    x1, x2, y = zip(*batch)
    return (
        pad_sequence(x1, batch_first=True, padding_value=padding_value),
        pad_sequence(x2, batch_first=True, padding_value=padding_value),
        torch.stack(y),
    )
