# proto_maml.py
"""
(c) Copyright 2025 by Ali Alfatemi
All rights reserved.

proto_maml.py
-------------
Defines the inner_update (fast adaptation) and meta_train_proto_maml (outer loop).
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim


def inner_update(model, support_x, support_y, lr_inner, steps):
    """
    Perform a few gradient steps on the support set (MAML style),
    using a combination of:
      - Prototypical consistency (MSE to class prototypes),
      - Distance-based classification loss (NLL).
    Returns a cloned 'adapted' model (fast weights).
    """
    new_model = copy.deepcopy(model)
    fast_optimizer = optim.SGD(new_model.parameters(), lr=lr_inner)

    for _ in range(steps):
        fast_optimizer.zero_grad()

        # 1) Forward pass on support
        emb_supp = new_model(support_x)

        # 2) Compute prototypes
        prototypes = []
        for c in [0, 1]:
            emb_c = emb_supp[support_y == c]
            if emb_c.size(0) == 0:
                proto = torch.zeros(emb_supp.shape[1], device=emb_supp.device)
            else:
                proto = emb_c.mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)  # (2, embed_dim)

        # 3) "Prototype consistency" loss
        proto_consistency = 0.0
        for c in [0, 1]:
            emb_c = emb_supp[support_y == c]
            if emb_c.size(0) > 0:
                proto_c = prototypes[c]
                proto_consistency += ((emb_c - proto_c) ** 2).mean()
        proto_consistency /= 2.0

        # 4) Distance-based classification on support
        dists = torch.cdist(emb_supp.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        log_p_y = torch.log_softmax(-dists, dim=1)
        ce_loss = nn.NLLLoss()(log_p_y, support_y)

        # Combined loss
        total_loss = proto_consistency + ce_loss
        total_loss.backward()
        fast_optimizer.step()

    return new_model


def meta_train_proto_maml(
    model, train_loader, outer_optimizer, lr_inner=1e-2, steps_inner=1, episodes=1000
):
    """
    Outer loop:
      - For each episode, do inner_update() on support
      - Evaluate on query => get loss => backprop to original model
    """
    model.train()
    all_losses, all_accs = [], []

    for episode_i, (support_x, support_y, query_x, query_y) in enumerate(
        train_loader, start=1
    ):
        # Squeeze out the batch dimension
        support_x = support_x.squeeze(0)
        support_y = support_y.squeeze(0)
        query_x = query_x.squeeze(0)
        query_y = query_y.squeeze(0)

        # 1) Adapt the model on support
        adapted_model = inner_update(
            model, support_x, support_y, lr_inner=lr_inner, steps=steps_inner
        )

        # 2) Evaluate on query
        emb_query = adapted_model(query_x)
        emb_supp = adapted_model(support_x)

        # Compute prototypes
        prototypes = []
        for c in [0, 1]:
            emb_c = emb_supp[support_y == c]
            if emb_c.size(0) == 0:
                proto = torch.zeros(emb_supp.shape[1], device=emb_supp.device)
            else:
                proto = emb_c.mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes, dim=0)

        dists = torch.cdist(emb_query.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)
        log_p_y = torch.log_softmax(-dists, dim=1)
        query_loss = nn.NLLLoss()(log_p_y, query_y)

        # Outer update
        outer_optimizer.zero_grad()
        grads = torch.autograd.grad(
            query_loss, adapted_model.parameters(), retain_graph=False
        )
        for param, g in zip(model.parameters(), grads):
            param.grad = g
        outer_optimizer.step()

        # Logging
        preds = torch.argmax(log_p_y, dim=1)
        query_acc = (preds == query_y).float().mean().item()

        all_losses.append(query_loss.item())
        all_accs.append(query_acc)

        if episode_i >= episodes:
            break

    return all_losses, all_accs
