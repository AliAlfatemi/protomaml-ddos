import copy

import numpy as np
import pytest
import torch
from protomaml.dataset import EpisodeDataset
from protomaml.model import ProtoMAMLNet
from protomaml.proto_maml import inner_update, meta_train_proto_maml
from protomaml.train_and_eval import evaluate_few_shot
from torch.utils.data import DataLoader

torch.set_num_threads(1)


def samples(n=30):
    return np.arange(n * 4, dtype=np.float32).reshape(n, 4) / 10, np.tile(
        [0, 1], n // 2
    )


def test_support_and_query_rows_never_overlap():
    X, y = samples()
    data = EpisodeDataset(X, y, k_shot=3, q_query=4)
    for _ in range(10):
        sx, sy, qx, qy = data[0]
        assert not set(sx[:, 0].tolist()) & set(qx[:, 0].tolist())
        assert len(sx) == 6 and len(qx) == 8
        assert sy.bincount().tolist() == [3, 3] and qy.bincount().tolist() == [4, 4]


@pytest.mark.parametrize("n,k,q", [(4, 2, 1), (8, 4, 4)])
def test_undersized_pools_are_rejected(n, k, q):
    X, y = samples(n)
    with pytest.raises(ValueError, match="distinct support and query"):
        EpisodeDataset(X, y, k_shot=k, q_query=q)


def test_inner_update_changes_copy_without_changing_initial_model():
    torch.manual_seed(1)
    model = ProtoMAMLNet(4)
    before = copy.deepcopy(model.state_dict())
    sx, sy, _, _ = EpisodeDataset(*samples(), k_shot=3, q_query=2)[0]
    adapted = inner_update(model, sx, sy, lr_inner=0.1, steps=2)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert any(not torch.equal(before[k], v) for k, v in adapted.state_dict().items())


def test_outer_update_reaches_initial_parameters():
    torch.manual_seed(1)
    model = ProtoMAMLNet(4)
    before = copy.deepcopy(model.state_dict())
    data = EpisodeDataset(*samples(), episodes=2, k_shot=3, q_query=2)
    losses, acc = meta_train_proto_maml(
        model,
        DataLoader(data, batch_size=1),
        torch.optim.Adam(model.parameters(), lr=0.01),
        episodes=2,
    )
    assert len(losses) == len(acc) == 2 and all(np.isfinite(losses))
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_evaluation_returns_one_prediction_per_query():
    X, y = samples()
    r = evaluate_few_shot(ProtoMAMLNet(4), X, y, episodes=3, k_shot=2, q_query=4)
    assert r["confusion_matrix"].sum() == 24
    assert 0 <= r["accuracy"] <= 1


def test_evaluation_rejects_overlapping_fallback():
    X, y = samples(4)
    with pytest.raises(ValueError, match="distinct support and query"):
        evaluate_few_shot(ProtoMAMLNet(4), X, y, k_shot=2, q_query=1)
