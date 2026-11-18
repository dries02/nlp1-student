import pandas as pd
import numpy as np
import gzip

from zipfile import ZipFile
from io import TextIOWrapper


def read_lines_from_file(path, encoding="utf-8"):
    if path.endswith('.txt'):
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith('.zip'):
        with ZipFile(path) as archive:
            members = archive.namelist()

            if len(members) != 1:
                raise ValueError(
                    f"Expected exactly one .txt file in {path}, "
                    f"found {len(members)}: {members}"
                )

            with archive.open(members[0]) as f:
                with TextIOWrapper(f, encoding=encoding) as text:
                    for line in text:
                        yield line.strip()
    else:
        raise ValueError(f"Unknown format: {path}")

class Vocabulary:
    """
    This is a helper class that wraps around a list of symbols/words/tokens and provides
    functions to map from str to int.
    """

    def __init__(self, init: list, default: str, frozen=False):
        if default not in init:
            raise ValueError("The default token must be in the init list")
        self.words = list(init)
        self._str2int = {w: i for i, w in enumerate(self.words)}
        self._default_word = default
        self._default_index = self._str2int[self._default_word]
        self._frozen = frozen

    @property
    def default_word(self):
        return self._default_word

    @property
    def default_index(self):
        return self._default_index

    def freeze(self):
        """After this operation add(.) no longer expands the vocabulary"""
        self._frozen = True

    def add(self, word: str) -> int:
        """
        Add a word to the vocabulary and return its id (existing words are not added again).
        If the vocabulary is frozen this adds nothing and returns the id of the default.
        """
        if self._frozen:
            return self.index(word)

        i = self._str2int.get(word, -1)
        if i < 0:
            i = len(self.words)
            self.words.append(word)
            self._str2int[word] = i
        return i

    def index(self, word: str) -> int:
        """Convert word to 0-based integer (0-based because Python lists are 0-based)"""
        return self._str2int.get(word, self.default_index)

    def word(self, index: int) -> str:
        """Convert an 0-based integer to a word"""
        if index >= len(self):  # if the index is out of bounds, we return the default str
            return self.default_word
        else:
            return self.words[index]

    def __contains__(self, word: str):
        return word in self._str2int

    def __len__(self):
        """Size of vocabulary"""
        return len(self.words)

    def __iter__(self):
        """Iterate over words"""
        return iter(self.words)

    def items(self):
        """Iterates over pairs of the kind (index, word) for all words in the vocabulary"""
        return enumerate(self.words)

    def __str__(self):
        return "\n".join(f"{i} {w}" for i, w in self.items())

class PretrainedEmbeddings:
    """
    Load an embedding matrix (stored in one of the supported files, see read_lines_from_file)
     and gives structured access to it.
    """

    def __init__(self, path, num_dims, unk_token="<unk>", pad_token="<pad>", encoding="utf-8", skip_first_line=False):
        """
        path: text/gz/zip file with embeddings
        num_dims: dimensionality of embeddings
        unk_token: special token reserve for words not in the table, we use the average of all embeddings as the unk embedding
        pad_token: special token used for batching sequences of variable length, we use a vector of zeros
        encoding: encoding of the embeddings file
        skip_first_line: some embedding files have metadata in the first line
        """
        vectors = [
            np.zeros(num_dims, dtype=np.float32),  # pad
            np.zeros(num_dims, dtype=np.float32) # unk placeholder
        ]
        self.vocab = Vocabulary(init=[pad_token, unk_token], default=unk_token)
        self.num_dims = num_dims
        self.unk_token = unk_token
        self.pad_token = pad_token

        for line_num, line in enumerate(read_lines_from_file(path, encoding=encoding)):
            if skip_first_line and line_num == 0:
                continue
            parts = line.split()
            word = parts[0].lower() # as we don't know much about their preprocessing, let's lowercase words here
            values = parts[1:]
            if len(values) != num_dims: # avoids some known problems with these files
                continue
            vector = np.array(values).astype(np.float32)
            if word in self.vocab:
                continue  # better skip clashes, if any
            self.vocab.add(word)
            vectors.append(vector)

        # [num_words, num_dims]
        vectors = np.stack(vectors, axis=0)
        # unk embedding is the mean of all others after pad (that's the "center" of the space)
        vectors[self.vocab.default_index] = np.mean(vectors[2:, :], 0)
        self.vectors = vectors
        # we freeze the vocab to avoid having it out of sync with the vectors
        self.vocab.freeze()
        assert self.vectors.shape[0] == len(self.vocab), "Mismatch between vocab and vectors"

    def __contains__(self, word: str):
        """Test whether or not a word (or its lowercased version) is in the vocabulary"""
        return word.lower() in self.vocab

    def index(self, word: str):
        """Return the id of a word (or of its lowercased version) or 0 (by definition, id of unk_token)"""
        return self.vocab.index(word.lower())

    def word(self, i: int):
        """Return the word associated with an id"""
        return self.vocab.word(i)

    def __getitem__(self, i: int):
        """Return self.vectors[i]"""
        return self.vectors[i]  # this will raise an exceptino if i is out of bounds

    def embed_word(self, word: str):
        """Return the vector representation of the word (or its lowercased version, or unk)"""
        i = self.index(word)
        return self.vectors[i]

    def embed_sequence(self, tokens: list):
        """Return a list of embeddings, one for each token in the input list"""
        return [self.embed_word(t) for t in tokens]


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all splits from Huggingface."""
    splits = {
        'train': 'labeled_final/train-00000-of-00001.parquet',
        'test': 'labeled_final/test-00000-of-00001.parquet',
        'validation': 'labeled_final/validation-00000-of-00001.parquet',
    }
    
    train_df = pd.read_parquet("hf://datasets/google-research-datasets/paws/" + splits["train"])
    dev_df = pd.read_parquet("hf://datasets/google-research-datasets/paws/" + splits["validation"])
    test_df = pd.read_parquet("hf://datasets/google-research-datasets/paws/" + splits["test"])

    return train_df, dev_df, test_df