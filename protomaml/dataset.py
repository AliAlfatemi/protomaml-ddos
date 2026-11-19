# dataset.py
"""
(c) Copyright 2025 by Ali Alfatemi
All rights reserved.

dataset.py
----------
Provides an EpisodeDataset for generating few-shot episodes (support, query sets).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class EpisodeDataset(Dataset):
    """
    Creates episodes for a 2-class few-shot scenario (BENIGN vs DDoS).
    Each __getitem__ returns (support_x, support_y, query_x, query_y).
    """
    def __init__(self, X, y, episodes=1000, k_shot=5, q_query=5):
        """
        Args:
            X: np.ndarray, shape (num_samples, num_features)
            y: np.ndarray, shape (num_samples,)
            episodes: how many episodes for this dataset
            k_shot: # of support samples per class
            q_query: # of query samples per class
        """
        super().__init__()
        self.X = X
        self.y = y
        self.episodes = episodes
        self.k_shot = k_shot
        self.q_query = q_query

        # Index by class
        if min(episodes, k_shot, q_query) < 1:
            raise ValueError("episode, support, and query counts must be positive")
        if len(X) != len(y) or set(np.unique(y)) != {0, 1}:
            raise ValueError("paired features and binary labels 0/1 are required")
        if any(np.sum(y == c) < k_shot + q_query for c in (0, 1)):
            raise ValueError("each class needs enough distinct support and query rows")
        self.indices_by_class = {}
        unique_labels = np.unique(self.y)
        for label in unique_labels:
            self.indices_by_class[label] = np.where(self.y == label)[0]

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        """
        Returns:
          support_x_t: shape [2*k_shot, num_features]
          support_y_t: shape [2*k_shot]
          query_x_t:   shape [2*q_query, num_features]
          query_y_t:   shape [2*q_query]
        """
        support_x_list, support_y_list = [], []
        query_x_list,   query_y_list   = [], []

        for class_label in [0, 1]:
            indices = self.indices_by_class[class_label]
            needed = self.k_shot + self.q_query

            # Sample with or without replacement
            if len(indices) < needed:
                chosen = np.random.choice(indices, size=needed, replace=False)
            else:
                chosen = np.random.choice(indices, size=needed, replace=False)

            s_idx = chosen[:self.k_shot]
            q_idx = chosen[self.k_shot:]

            support_x_list.append(self.X[s_idx])
            support_y_list.append([class_label]*self.k_shot)

            query_x_list.append(self.X[q_idx])
            query_y_list.append([class_label]*self.q_query)

        support_x = np.concatenate(support_x_list, axis=0)
        support_y = np.concatenate(support_y_list, axis=0)
        query_x   = np.concatenate(query_x_list, axis=0)
        query_y   = np.concatenate(query_y_list, axis=0)

        # Convert to tensors
        support_x_t = torch.tensor(support_x, dtype=torch.float)
        support_y_t = torch.tensor(support_y, dtype=torch.long)
        query_x_t   = torch.tensor(query_x,   dtype=torch.float)
        query_y_t   = torch.tensor(query_y,   dtype=torch.long)

        return support_x_t, support_y_t, query_x_t, query_y_t
