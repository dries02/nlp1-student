# Paraphrase Detection:
Each data point is a triplet $(x_1,x_2,y)\in\Sigma^\ast\times\Sigma^\ast\times\{0,1\}$, where $x_1$ and $x_2$ are sentences, and $y=1$ iff they are paraphrases. We use three feature extraction methods $h$, which are encoders:
1. With a token indicator we map $h:\Sigma^\ast\to\{0,1\}^D$. This counts presence/absence.
2. With a bag-of-words, we map $h:\Sigma^\ast\to\mathbb{N}^D$. This counts the number of occurrences.
3. With a sentence transformer, we map $h:\Sigma^\ast\to\mathbb{R}^D$. This is a high-dimensional embedding of a pretrained encoder.

Some remarks:
- We ensure that (1) and (2) fit on both $x_1$ and $x_2$ together. We do not fit (3) as it is pretrained.
- encoder $\neq$ tokenizer

We obtain the feature vectors $h(x_1)$ and $h(x_2)$.



