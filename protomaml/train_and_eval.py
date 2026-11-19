# train_and_eval.py
"""
(c) Copyright 2025 by Ali Alfatemi
All rights reserved.

train_and_eval.py
-----------------
High-level functions for training (train_few_shot) and evaluating (evaluate_few_shot).
Now includes a logging mechanism to return the train-loss and train-accuracy curves.
"""

import numpy as np
import torch

from .proto_maml import meta_train_proto_maml, inner_update
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    cohen_kappa_score, matthews_corrcoef
)

def train_few_shot(
    model,
    train_loader,
    optimizer,
    lr_inner=1e-2,
    steps_inner=1,
    episodes=1000
):
    """
    Wraps meta_train_proto_maml. Returns the history of (losses, accs).
    This is typically for plotting the training curve.
    """
    losses, accs = meta_train_proto_maml(
        model=model,
        train_loader=train_loader,
        outer_optimizer=optimizer,
        lr_inner=lr_inner,
        steps_inner=steps_inner,
        episodes=episodes
    )
    return losses, accs

def evaluate_few_shot(
    model,
    X_test,
    y_test,
    lr_inner=1e-2,
    steps_inner=1,
    episodes=100,
    k_shot=5,
    q_query=5,
    device="cpu"
):
    """
    1) For each 'episodes' iteration, sample a new support+query from test set.
    2) Inner-update on support with grad => adapted model.
    3) Evaluate on query => collect predictions.
    4) Return extended metrics (accuracy, precision, recall, F1, confusion_matrix, classification_report,
       cohen_kappa, mcc).
    """
    if min(episodes, k_shot, q_query, steps_inner) < 1:
        raise ValueError("episode and adaptation counts must be positive")
    if any(np.sum(y_test == c) < k_shot + q_query for c in (0, 1)):
        raise ValueError("each test class needs distinct support and query rows")
    model.eval()

    idx_benign = np.where(y_test == 0)[0]
    idx_ddos   = np.where(y_test == 1)[0]

    all_true = []
    all_pred = []

    for _ in range(episodes):
        # Sample an episode
        s_x_list, s_y_list = [], []
        q_x_list, q_y_list = [], []
        for c in [0, 1]:
            indices = idx_benign if c == 0 else idx_ddos
            needed = k_shot + q_query

            if len(indices) < needed:
                chosen = np.random.choice(indices, needed, replace=False)
            else:
                chosen = np.random.choice(indices, needed, replace=False)

            s_idx = chosen[:k_shot]
            q_idx = chosen[k_shot:]

            s_x_list.append(X_test[s_idx])
            s_y_list.append([c]*k_shot)

            q_x_list.append(X_test[q_idx])
            q_y_list.append([c]*q_query)

        support_x = np.concatenate(s_x_list, axis=0)
        support_y = np.concatenate(s_y_list, axis=0)
        query_x   = np.concatenate(q_x_list, axis=0)
        query_y   = np.concatenate(q_y_list, axis=0)

        # Convert to torch
        support_x_t = torch.tensor(support_x, dtype=torch.float).to(device)
        support_y_t = torch.tensor(support_y, dtype=torch.long).to(device)
        query_x_t   = torch.tensor(query_x,  dtype=torch.float).to(device)
        query_y_t   = torch.tensor(query_y,  dtype=torch.long).to(device)

        # Inner update
        adapted_model = inner_update(
            model,
            support_x_t,
            support_y_t,
            lr_inner=lr_inner,
            steps=steps_inner
        )

        # Evaluate on query
        with torch.no_grad():
            emb_query = adapted_model(query_x_t)
            emb_supp  = adapted_model(support_x_t)

            prototypes = []
            for c in [0,1]:
                emb_c = emb_supp[support_y_t == c]
                if emb_c.size(0) == 0:
                    proto = torch.zeros(emb_supp.shape[1], device=emb_supp.device)
                else:
                    proto = emb_c.mean(dim=0)
                prototypes.append(proto)
            prototypes = torch.stack(prototypes, dim=0)

            dists = torch.cdist(emb_query.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
            log_p_y = torch.log_softmax(-dists, dim=1)

            preds = torch.argmax(log_p_y, dim=1).cpu().numpy()

        all_true.append(query_y)
        all_pred.append(preds)

    # Combine from all episodes
    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)

    acc  = (y_true_all == y_pred_all).mean()
    prec = precision_score(y_true_all, y_pred_all, average='binary', zero_division=0)
    rec  = recall_score(y_true_all, y_pred_all, average='binary', zero_division=0)
    f1   = f1_score(y_true_all, y_pred_all, average='binary', zero_division=0)
    cm   = confusion_matrix(y_true_all, y_pred_all)
    c_report = classification_report(y_true_all, y_pred_all, target_names=["BENIGN", "DDoS"], zero_division=0)

    # Additional
    kappa = cohen_kappa_score(y_true_all, y_pred_all)
    mcc   = matthews_corrcoef(y_true_all, y_pred_all)

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": c_report,
        "cohen_kappa": kappa,
        "mcc": mcc
    }

    return results
