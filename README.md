# DeepSched

DeepSched is a research implementation of a deep learning-assisted CPU workload
prediction and scheduling framework. The goal is to move beyond purely reactive
CPU scheduling heuristics by forecasting near-future CPU utilization and using
those forecasts to guide scheduling decisions.

The project is being built step by step so that the final repository contains
the code, experimental results, figures, and reproducibility material needed for
a complete research paper.

## Development Stages

DeepSched is being developed in two stages.

### Stage 1: Prototype Development

Stage 1 builds the complete end-to-end pipeline on a compact Alibaba-derived
dataset sample. The goal is to make every component work before scaling up:

- data preprocessing
- workload prediction
- model ablations
- forecast quality analysis
- scheduler simulation
- forecast-aware rule scheduling
- PPO scheduling
- figure generation
- Stage 1 summary and conclusion

Stage 1 results are meaningful for validating the method and implementation, but
they are not final paper-level generalization claims.

### Stage 2: Full Experimental Development

Stage 2 will expand the datasets and rerun the full pipeline. It will use
broader Alibaba coverage and Google Cluster Trace data, then repeat all
forecasting, scheduling, PPO, ablation, and figure-generation steps.

Stage 2 is where the final research-paper claims should be made.

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

The repository is organized around Stage 1 research milestones:

| Milestone | Purpose | Expected Outputs |
| --- | --- | --- |
| C-01 | Data download and preprocessing | `data/processed/*.pt`, `scaler.pkl`, `figures/data_eda.pdf` |
| C-02 | LSTM workload prediction baseline | baseline model, MAE/RMSE/MAPE results |
| C-03 | CNN-LSTM-Attention WPM | main forecasting model |
| C-04 | WPM ablations | forecasting comparison table and plots |
| C-05 | CPU scheduling simulator | Gymnasium environment |
| C-06 | Classical schedulers and rule advisor | scheduling baseline table |
| C-07 | PPO scheduling agent | trained RL scheduler and reward curves |
| C-08 | Paper figures | final publication-quality figures |
| C-09 | Stage 1 report | current results, conclusion, Stage 2 plan |

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

### Completed: C-03 CNN-LSTM-Attention WPM

The main workload prediction model has been added. It uses a 1-D CNN front-end
for local pattern extraction, stacked LSTM layers for temporal modeling, and
multi-head attention for weighting relevant historical time steps.

Current quick benchmark results:

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| LSTM baseline | 0.0767 | 0.1015 | 19.64% |
| CNN-LSTM-Attention | 0.0744 | 0.0977 | 19.12% |

In this first run, the CNN-LSTM-Attention model improves over the LSTM baseline
on MAE and RMSE. MAPE is also lower than the LSTM baseline in the C-04
architecture ablation table below.

Generated files:

- `figures/cnn_lstm_attn_forecast.pdf`
- `figures/cnn_lstm_attn_attention_weights.pdf`

### Completed: C-04 WPM Ablation Studies

Architecture and horizon ablations have been implemented. The architecture
ablation isolates the effect of CNN feature extraction and attention. The
horizon ablation compares short and longer forecast horizons supported by the
current processed tensor target.

Current architecture ablation results:

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| LSTM only | 0.0778 | 0.1022 | 20.09% |
| CNN-LSTM | 0.0751 | 0.0991 | 19.31% |
| LSTM-Attention | 0.0772 | 0.1016 | 19.81% |
| CNN-LSTM-Attention | 0.0744 | 0.0977 | 19.12% |

Current horizon ablation results:

| Horizon | MAE | RMSE | MAPE |
| ---: | ---: | ---: | ---: |
| H=1 | 0.0620 | 0.0840 | 15.32% |
| H=5 | 0.0698 | 0.0923 | 18.27% |
| H=10 | 0.0759 | 0.1005 | 18.89% |

The current results show the expected trend: shorter horizons are easier to
predict, and the full CNN-LSTM-Attention model is the strongest architecture in
this quick ablation run.

Generated files:

- `figures/ablation_bar.pdf`
- `figures/horizon_line.pdf`
- `figures/lstm_only_forecast.pdf`
- `figures/cnn_lstm_forecast.pdf`
- `figures/lstm_attention_forecast.pdf`
- `figures/lstm_attention_attention_weights.pdf`

### Completed: C-04.5 Forecast Quality Analysis

Aggregate forecasting metrics are not enough for a scheduling paper, so the
repository now includes additional forecast-quality analysis:

- error by forecast horizon step
- burst detection quality
- peak detection quality

Current finding:

- CNN-LSTM-Attention improves horizon-wise RMSE over the LSTM baseline.
- Burst and peak detection are still weak on the small current dataset.
- This confirms that the final paper needs larger Alibaba coverage plus Google
  Cluster Trace experiments before making strong generalization claims.

Generated files:

- `results/forecast_horizon_metrics.csv`
- `results/forecast_event_metrics.csv`
- `figures/forecast_error_vs_horizon.pdf`
- `figures/burst_peak_detection.pdf`

### Completed: C-05 Gymnasium Scheduler Environment

The project now includes a Gymnasium-compatible CPU scheduling environment for
the reinforcement-learning stage. The environment simulates a bounded ready
queue, process arrivals, CPU burst execution, time-quantum scheduling, and
episode-level scheduling metrics.

Environment design:

- observation: normalized remaining bursts, waiting times, and optional WPM
  forecast vector
- action: select one ready-queue slot to run
- reward: negative weighted cost of waiting, energy proxy, and queue pressure
- metrics: AWT, ATT, throughput, CPU utilization, energy proxy, completed jobs

Validation:

- `env/validate_env.py` passes Gymnasium's environment checker.
- `scheduler/random_agent.py` produces a random-agent sanity baseline.

Generated file:

- `results/random_agent.csv`

### Completed: C-06 First Scheduling Metrics

After validating the scheduler environment milestone, the project now includes
classical scheduling baselines and a forecast-aware rule advisor.

Current scheduling comparison:

| Scheduler | AWT | ATT | Throughput |
| --- | ---: | ---: | ---: |
| FCFS | 18.47 | 34.43 | 0.0333 |
| SJF | 13.51 | 29.47 | 0.0333 |
| RR-20 | 19.19 | 35.15 | 0.0333 |
| Rule-WPM | 11.37 | 27.33 | 0.0333 |

The current Rule-WPM advisor uses predicted load level and forecast slope. Under
high or rising predicted utilization it chooses the shortest remaining job;
otherwise it keeps FCFS behavior. This gives the project an interpretable
forecast-aware scheduling result even before PPO is added.

Generated files:

- `results/scheduling_metrics.csv`
- `figures/baseline_comparison.pdf`

### Completed: C-07 PPO Scheduler

Stage 1 PPO training has been implemented using Stable-Baselines3. Two agents
are trained and evaluated in the Gymnasium scheduling environment:

- `PPO_no_forecast`: policy observes queue state without WPM forecasts
- `PPO_DeepSched`: policy observes queue state plus WPM forecast features

Current Stage 1 RL comparison inside `CPUSchedEnv`:

| Scheduler | AWT | ATT | Mean Reward |
| --- | ---: | ---: | ---: |
| Random | 155.10 | 172.01 | -0.4424 |
| FCFS policy | 153.45 | 170.17 | -0.4407 |
| SJF policy | 84.43 | 101.23 | -0.6650 |
| Rule-WPM policy | 84.31 | 101.11 | -0.6614 |
| PPO no forecast | 75.83 | 92.69 | -0.2973 |
| PPO DeepSched | 89.06 | 105.63 | -0.3321 |

Important Stage 1 finding: PPO learns a useful scheduling policy compared with
random and fixed heuristic policies, but the forecast-aware PPO is not yet
better than PPO without forecasts. This means PPO needs further reward/state
tuning in Stage 2; the forecast-aware rule advisor remains the stronger
interpretable scheduling result for now.

Generated files:

- `models/PPO_no_forecast_stage1.zip`
- `models/PPO_DeepSched_stage1.zip`
- `results/ppo_metrics.csv`
- `results/ppo_training_curve.csv`

### Completed: C-08 Figure Builder

The repository now includes a single script for rebuilding paper summary
figures from saved result CSV files.

Generated or rebuilt files:

- `figures/ablation_bar.pdf`
- `figures/horizon_line.pdf`
- `figures/forecast_error_vs_horizon.pdf`
- `figures/burst_peak_detection.pdf`
- `figures/ppo_vs_baselines.pdf`
- `figures/ppo_reward_curve.pdf`

### Completed: C-09 Stage 1 Report and Conclusion

The Stage 1 summary has been written in `docs/stage1_report.md`. It collects
the current results, explains what they mean, documents the key limitations, and
defines the Stage 2 plan.

Current Stage 1 conclusion:

- DeepSched is implemented end to end.
- CNN-LSTM-Attention is the strongest Stage 1 forecasting model.
- Forecast-aware Rule-WPM is the strongest interpretable scheduling result.
- PPO is operational but requires Stage 2 reward/state tuning.
- Larger Alibaba and Google datasets are required before final paper claims.

## How to Reproduce the Current Results

Install dependencies:

```powershell
pip install -r requirements.txt
```

GPU note: PyTorch automatically uses CUDA when available. On the current
development machine, CUDA is available with one `NVIDIA GeForce RTX 4050 Laptop
GPU`, so model training uses the GPU by default through `train.get_device()`.

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

Train and evaluate the CNN-LSTM-Attention WPM:

```powershell
python train.py --model wpm --epochs 20 --batch-size 64 --hidden-size 64 --cnn-channels 32 --n-heads 4
```

Rebuild CNN-LSTM-Attention metrics and figures from the saved checkpoint:

```powershell
python scripts/evaluate_wpm.py
```

Run forecast quality analysis:

```powershell
python analysis/forecast_quality.py
```

Run C-04 ablations:

```powershell
python scripts/run_ablation.py --epochs 15 --batch-size 64 --hidden-size 64 --cnn-channels 32
```

Run scheduling baselines and Rule-WPM:

```powershell
python scheduler/run_baselines.py
```

Validate the Gymnasium scheduling environment:

```powershell
python env/validate_env.py
```

Run the random-agent environment baseline:

```powershell
python scheduler/random_agent.py
```

Train and evaluate Stage 1 PPO agents:

```powershell
python scheduler/ppo_agent.py --total-timesteps 12000 --n-envs 4 --eval-seeds 100 101
```

Re-evaluate PPO and same-environment heuristic baselines:

```powershell
python scheduler/evaluate_rl.py
```

Rebuild C-08 summary figures:

```powershell
python figures/plot_all.py
```

Read the C-09 Stage 1 report:

```powershell
Get-Content docs/stage1_report.md
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
  docs/
    stage1_report.md       # C-09 Stage 1 report and conclusion
  data/
    preprocess.py          # C-01 preprocessing pipeline
  scripts/
    download_data.py       # dataset sample downloader
  figures/                 # generated research figures, ignored by Git
  models/                  # trained models, ignored by Git
  results/                 # metric CSV files, ignored by Git
  requirements.txt
```

## Stage 2 Plan

After Stage 1, the next development phase will:

- expand Alibaba trace coverage beyond the compact sample
- add Google Cluster Trace experiments
- rerun preprocessing for each dataset
- retrain all WPM models and ablations
- rerun forecast-quality analysis, including burst and peak detection
- recalibrate Rule-WPM thresholds using validation data
- tune PPO state representation, reward design, and training length
- evaluate PPO with and without forecasts across more seeds
- rebuild all final paper figures and tables
- write the full research paper from the Stage 2 results

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
