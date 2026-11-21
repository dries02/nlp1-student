import torch
from nltk.tokenize import TreebankWordTokenizer

_tokenizer = TreebankWordTokenizer()


def tokenize_split(text, model):
    """Treebank tokens without punctuation; hyphenated words the model doesn't know are split."""
    out = []
    for token in _tokenizer.tokenize(text):
        if "-" in token and token not in model:
            out.extend(p for p in token.split("-") if p)
        else:
            out.append(token)
    return [t for t in out if any(c.isalnum() for c in t)]


def tokenize_df(df, model):
    """Tokenizes both sentence columns of a PAWS dataframe."""
    tokens1 = [tokenize_split(s, model) for s in df["sentence1"]]
    tokens2 = [tokenize_split(s, model) for s in df["sentence2"]]
    return tokens1, tokens2


def build_reduced_embeddings(token_lists, model):
    """
    Keeps only the vectors of words that occur in our data about 32k instead of 1.29M words.
    Results are the same for frozen embeddings, but training is much faster and uses less memory.
    Returns the smaller embedding matrix and a map from old ids to new ids (0 = pad, 1 = unk stay the same).
    """
    full_ids = sorted({model.index(t) for tokens in token_lists for t in tokens} - {0, 1})
    keep = [0, 1] + full_ids
    remap = {full_id: new_id for new_id, full_id in enumerate(keep)}
    matrix = torch.tensor(model.vectors[keep], dtype=torch.float32)
    return matrix, remap


def to_ids(tokens, model, remap):
    """Converts tokens to ids in the reduced embedding matrix (empty sentences get a single unk)."""
    ids = [remap[model.index(t)] for t in tokens]
    return ids if ids else [1]
