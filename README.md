# ProtoMAML for Few-Shot DDoS Detection

[![CI](https://github.com/AliAlfatemi/protomaml-ddos/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAlfatemi/protomaml-ddos/actions/workflows/ci.yml)

Episodic binary classification with learned embeddings, support-set prototypes,
inner-loop adaptation, and first-order outer updates. Includes the original
model and adaptation implementation, a portable CPU command, and synthetic tests.

**Paper:** [ProtoMAML: A Hybrid Meta-Learning Approach Integrating Prototypical Networks for Data-Efficient DDoS Attack Detection](https://doi.org/10.1109/IWCMC65282.2025.11059701)

**Authors:** Ali Alfatemi, Mohamed Rahouti, Zakirul Alam Bhuiyan, Aiman Solyman, Mohammed Aledhari

**Venue:** International Wireless Communications and Mobile Computing, 2025.

## Install and check

```bash
git clone https://github.com/AliAlfatemi/protomaml-ddos.git
cd protomaml-ddos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m protomaml smoke
```

## Train and evaluate episodes

Provide a numeric-feature CSV with a `Label` column. By default `BENIGN` is
class 0 and other labels are class 1. Use `--benign-label 0` for 0/1 labels.

```bash
python -m protomaml train --csv /path/to/prepared-flows.csv \
  --output outputs/run-001 --episodes 200 --eval-episodes 100 \
  --k-shot 5 --q-query 5
```

Rows are split into independent training and evaluation pools before fitting
the scaler. Each episode samples distinct support and query rows within its
pool. Both classes need at least `k_shot + q_query` rows in each pool; the
release rejects undersized pools instead of sampling overlapping examples.

The command writes local weights, fitted scaler, metadata, and episode metrics.
Output directories must be new. Inputs, model weights, results, and manuscript
files are excluded from Git. See `--help` for adaptation and seed controls.

## Method and release scope

The retained learner uses a copied model for support adaptation and transfers
query gradients back to the initial model for a **first-order** update. It does
not differentiate through the complete inner optimization trajectory.
The release adds packaging, a train-only scaling path, and disjoint episode
validation. It does not claim that synthetic checks reproduce published results.
The historical globally fitted scaler and replacement-based episode fallback
are not used by the command. Real-data protocol replication remains separate.

## Development

```bash
python -m pytest -q
python -m build
```

CI runs tests, an episodic smoke check, and an installed-wheel check on Python
3.12 and 3.13. `protomaml/` contains the implementation; `tests/` uses synthetic
inputs. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation and license

[CITATION.cff](CITATION.cff) preserves the published paper's complete author
order. Paper authorship and software contribution credit are distinct.
The original restrictive [LICENSE](LICENSE) is preserved; public source
availability does not grant additional reuse permissions.
