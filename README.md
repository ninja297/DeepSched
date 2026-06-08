# DeepSched

DeepSched is a research implementation of a deep learning-assisted CPU workload
prediction and scheduling framework. The goal is to move beyond purely reactive
CPU scheduling heuristics by forecasting near-future CPU utilization and using
those forecasts to guide scheduling decisions.

The project is being built step by step so that the final repository contains
the code, experimental results, figures, and reproducibility material needed for
a complete research paper.

## Research Goal

Traditional CPU schedulers such as FCFS, SJF, Round Robin, and CFS make
decisions from the current system state. They do not explicitly predict future
workload behavior, which can lead to higher waiting time, turnaround time, and
energy use under bursty or dynamic workloads.

DeepSched proposes a two-stage pipeline:

1. Workload Prediction Module (WPM)
   - Learns CPU utilization patterns from historical trace data.
   - Starts with an LSTM baseline.
   - Extends to a CNN-LSTM-Attention model for stronger short-horizon
     forecasting.

2. Scheduling Advisor Module (SAM)
   - Uses predicted CPU utilization to guide scheduling decisions.
   - Starts with classical baselines and a rule-based advisor.
   - Extends to a PPO reinforcement learning agent in a simulated CPU
     scheduling environment.

The final evaluation will compare DeepSched against classical schedulers and
learning-based baselines using prediction and scheduling metrics.

## Implementation Plan

The repository is organized around research milestones:

| Stage | Purpose | Expected Outputs |
| --- | --- | --- |
| C-01 | Data download and preprocessing | `data/processed/*.pt`, `scaler.pkl`, `figures/data_eda.pdf` |
| C-02 | LSTM workload prediction baseline | baseline model, MAE/RMSE/MAPE results |
| C-03 | CNN-LSTM-Attention WPM | main forecasting model |
| C-04 | WPM ablations | forecasting comparison table and plots |
| C-05 | CPU scheduling simulator | Gymnasium environment |
| C-06 | Classical schedulers and rule advisor | scheduling baseline table |
| C-07 | PPO scheduling agent | trained RL scheduler and reward curves |
| C-08 | Paper figures | final publication-quality figures |

## Current Status

### Completed: C-01 Data Pipeline

The first milestone is implemented and verified.

Current pipeline:

- Downloads a compact public Alibaba 2018 machine-usage sample.
- Reads CPU utilization sampled at 300-second intervals.
- Normalizes the time series.
- Builds sliding-window tensors with:
  - lookback window `L = 60`
  - forecast horizon `H = 10`
  - feature count `F = 1`
- Saves train/validation/test splits for PyTorch models.
- Generates an exploratory data-analysis figure.

Verified tensor shapes:

| Split | X Shape | y Shape |
| --- | --- | --- |
| Train | `(1521, 60, 1)` | `(1521, 10, 1)` |
| Validation | `(326, 60, 1)` | `(326, 10, 1)` |
| Test | `(327, 60, 1)` | `(327, 10, 1)` |

Generated files:

- `data/processed/train.pt`
- `data/processed/val.pt`
- `data/processed/test.pt`
- `data/processed/scaler.pkl`
- `figures/data_eda.pdf`

Large raw and generated files are intentionally ignored by Git so the repository
stays lightweight and reproducible.

### Completed: C-02 LSTM Baseline

The first workload prediction baseline has been implemented using a stacked LSTM
model. It predicts the next 10 CPU-utilization steps from the previous 60 steps.

Current baseline configuration:

- model: 2-layer LSTM
- hidden size: 64 for the first quick benchmark run
- optimizer: Adam
- loss: mean squared error
- epochs: 20

Current test-set results:

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| LSTM baseline | 0.0767 | 0.1015 | 19.64% |

Generated files:

- `results/wpm_metrics.csv`
- `figures/lstm_baseline_forecast.pdf`

## How to Reproduce the Current Results

Install dependencies:

```powershell
pip install -r requirements.txt
```

Download the current default dataset sample:

```powershell
python scripts/download_data.py
```

Run preprocessing:

```powershell
python data/preprocess.py
```

This recreates the processed tensors and the first EDA figure.

Train and evaluate the current LSTM baseline:

```powershell
python train.py --model lstm --epochs 20 --batch-size 64 --hidden-size 64
```

Rebuild LSTM baseline metrics and forecast figure from the saved checkpoint:

```powershell
python scripts/evaluate_lstm_baseline.py
```

## Dataset Sources

The proposal targets two real-world cluster workload datasets:

- Alibaba Cluster Trace 2018
- Google Cluster Trace

For the first runnable milestone, the repository uses a compact Zenodo-hosted
Alibaba 2018 machine-usage sample derived from the original Alibaba trace. This
keeps the first pipeline lightweight while still using real workload data.

Google Cluster Trace support has also been started. A starter Google
`task_usage` shard can be downloaded with:

```powershell
python scripts/download_data.py --source google
```

Google preprocessing support is included, but larger Google trace coverage will
be added later because a single shard does not provide enough time steps for the
default 60-step lookback window.

## Repository Structure

```text
DeepSched/
  data/
    preprocess.py          # C-01 preprocessing pipeline
  scripts/
    download_data.py       # dataset sample downloader
  figures/                 # generated research figures, ignored by Git
  models/                  # trained models, ignored by Git
  results/                 # metric CSV files, ignored by Git
  requirements.txt
  DeLTA26_048_Proposal.pdf
```

## Planned Updates

Next updates will add:

- CNN-LSTM-Attention workload prediction model.
- Prediction plots comparing actual and forecasted utilization.
- Ablation experiments for model architecture and forecast horizon.
- CPU scheduling simulator using Gymnasium.
- Classical scheduling baselines: FCFS, SJF, and Round Robin.
- WPM-guided rule-based scheduling advisor.
- PPO reinforcement learning scheduler.
- Final result tables and paper-ready figures.

## Expected Paper Results

The final repository should produce:

- Table I: workload prediction comparison across LSTM, CNN-LSTM, attention, and
  ablation variants.
- Table II: scheduling comparison across classical schedulers, rule-based
  DeepSched, PPO without forecasts, and PPO with forecasts.
- Figures for dataset analysis, prediction performance, attention weights,
  scheduling comparison, and PPO training convergence.

These outputs will support the final research paper with reproducible code,
trace-based experiments, and documented results.
