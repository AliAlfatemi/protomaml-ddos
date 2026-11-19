# model.py
"""
(c) Copyright 2025 by Ali Alfatemi
All rights reserved.

model.py
--------
Defines the ProtoMAMLNet, a simple feedforward embedding model.
"""

import torch.nn as nn
import torch.nn.functional as F


class ProtoMAMLNet(nn.Module):
    """
    Simple feedforward network to produce embeddings for few-shot classification.
    """

    def __init__(self, input_dim, embed_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
