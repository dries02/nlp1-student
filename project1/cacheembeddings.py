import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.utils import load_datasets

def main():
    print("Loading datasets")
    train_df, dev_df, test_df = load_datasets()
    
    # Reset index so aligns with array indices
    all_train = pd.concat([train_df, dev_df]).reset_index(drop=True)

    print("Loading SentenceTransformer model")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # encode for dev and train
    print("Encoding Train+Dev sentence 1")
    emb_train_s1 = model.encode(all_train["sentence1"].tolist(), show_progress_bar=True, batch_size=64)
    np.save("embeddings/train_dev_s1.npy", emb_train_s1)

    print("Encoding Train+Dev sentence 2")
    emb_train_s2 = model.encode(all_train["sentence2"].tolist(), show_progress_bar=True, batch_size=64)
    np.save("embeddings/train_dev_s2.npy", emb_train_s2)

    # Encode test set
    print("Encoding Test sentence 1")
    emb_test_s1 = model.encode(test_df["sentence1"].tolist(), show_progress_bar=True, batch_size=64)
    np.save("embeddings/test_s1.npy", emb_test_s1)

    print("Encoding Test sentence 2")
    emb_test_s2 = model.encode(test_df["sentence2"].tolist(), show_progress_bar=True, batch_size=64)
    np.save("embeddings/test_s2.npy", emb_test_s2)

    print("Done")

if __name__ == "__main__":
    main()