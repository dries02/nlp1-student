"""Training, evaluation and cross-validation for the pair classifiers."""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from .data import collate_pairs
from .models import EmbedAndPoolEncoder, RNNEncoder, PairClassifier, JointClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, embedding_matrix):
    matrix = embedding_matrix.clone()
    if cfg.get("random_init", False):
        matrix = torch.randn_like(matrix) * embedding_matrix[2:].std()
        matrix[0] = 0 

    encoder_type = cfg.get("encoder", "pool")  
    if encoder_type == "pool":
        encoder = EmbedAndPoolEncoder(matrix, pooling=cfg["pooling"], freeze_embeddings=cfg["freeze"])
    else:
        encoder = RNNEncoder(
            matrix, cell=encoder_type, hidden_size=cfg["rnn_hidden"], bidirectional=cfg["bidirectional"],
            num_layers=cfg.get("rnn_layers", 1), dropout=cfg["dropout"], pooling=cfg["rnn_pooling"],
            freeze_embeddings=cfg["freeze"],
        )

    if cfg.get("joint", False):
        return JointClassifier(encoder, hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
                               dropout=cfg["dropout"])
    return PairClassifier(encoder, combine=cfg["combine"], hidden_size=cfg["hidden_size"],
                          num_layers=cfg["num_layers"], dropout=cfg["dropout"])


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x1, x2, _ in loader:
            preds.append(model(x1.to(device), x2.to(device)).argmax(dim=-1).cpu())
    return torch.cat(preds).numpy()


def accuracy(model, dataset, idx, device):
    loader = DataLoader(Subset(dataset, idx), batch_size=512, shuffle=False, collate_fn=collate_pairs)
    preds = predict(model, loader, device)
    return float((preds == dataset.y[idx].numpy()).mean()), preds


def train_and_evaluate(dataset, train_idx, val_idx, embedding_matrix, cfg, device="cpu", verbose=True):
    """
    Trains one model on train_idx and returns (accuracy on val_idx, predictions on val_idx, history per epoch).

    Optional settings in cfg:
    - train_fraction (default 1.0): use only part of the training data (to study the effect of data size)
    - early_stopping (default False): hold out 10% of the training data as a dev set, keep the epoch with
      the best dev accuracy and stop after `patience` epochs without improvement. The validation fold is
      never used for these decisions.
    """
    set_seed(cfg["seed"])
    labels = dataset.y.numpy()
    train_idx = np.asarray(train_idx)

    if cfg.get("train_fraction", 1.0) < 1.0:
        train_idx, _ = train_test_split(train_idx, train_size=cfg["train_fraction"],
                                        stratify=labels[train_idx], random_state=cfg["seed"])
    dev_idx = None
    if cfg.get("early_stopping", False):
        train_idx, dev_idx = train_test_split(train_idx, test_size=0.1,
                                              stratify=labels[train_idx], random_state=cfg["seed"])

    model = build_model(cfg, embedding_matrix).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg["batch_size"],
                              shuffle=True, collate_fn=collate_pairs, generator=generator)

    history = []
    best_dev, best_state, best_epoch, epochs_without_improvement = -1.0, None, 0, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x1, x2), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        entry = {"epoch": epoch + 1, "train_loss": total_loss / len(train_idx)}
        entry["val_accuracy"], _ = accuracy(model, dataset, val_idx, device)  # only logged, never used for decisions
        if dev_idx is not None:
            entry["dev_accuracy"], _ = accuracy(model, dataset, dev_idx, device)
        history.append(entry)
        if verbose:
            dev_text = f", dev acc {entry['dev_accuracy']:.4f}" if dev_idx is not None else ""
            print(f"    epoch {epoch + 1}: train loss {entry['train_loss']:.4f}{dev_text}, val acc {entry['val_accuracy']:.4f}")

        if dev_idx is not None:
            if entry["dev_accuracy"] > best_dev:
                best_dev, best_epoch, epochs_without_improvement = entry["dev_accuracy"], epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.get("patience", 3):
                    if verbose:
                        print(f"    early stopping, best dev epoch {best_epoch}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    acc, preds = accuracy(model, dataset, val_idx, device)
    return acc, preds, history


def cross_validate(dataset, folds, embedding_matrix, cfg, device="cpu"):
    """Runs train_and_evaluate on every fold. Predictions are stored for every pair (each pair is in one validation fold)."""
    accuracies, histories = [], []
    predictions = np.full(len(dataset), -1)
    for k, (train_idx, val_idx) in enumerate(folds):
        print(f"  fold {k + 1}/{len(folds)}")
        fold_cfg = {**cfg, "seed": cfg["seed"] + k}
        acc, preds, history = train_and_evaluate(dataset, train_idx, val_idx, embedding_matrix, fold_cfg, device)
        predictions[val_idx] = preds
        accuracies.append(acc)
        histories.append(history)
        print(f"  fold {k + 1} accuracy: {acc:.4f}")
    return {
        "config": cfg,
        "accuracies": accuracies,
        "mean_accuracy": float(np.mean(accuracies)),
        "predictions": predictions.tolist(),
        "histories": histories,
    }
