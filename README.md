# DeepSched

Deep learning for CPU workload prediction and intelligent process scheduling.

This repository implements the project described in `DeLTA26_048_Proposal.pdf`.
The first milestone is the C-01 data pipeline:

- download a compact public Alibaba 2018 machine-usage sample
- optionally aggregate Google task CPU usage into 5-minute machine-level series
- normalize and create sliding-window tensors
- save train/validation/test splits for workload prediction
- generate an exploratory data-analysis figure

## Quick Start

```powershell
python scripts/download_data.py
python data/preprocess.py
```

Expected C-01 outputs:

- `data/processed/train.pt`
- `data/processed/val.pt`
- `data/processed/test.pt`
- `data/processed/scaler.pkl`
- `figures/data_eda.pdf`

## Dataset Notes

The proposal targets Google Cluster Trace and Alibaba Cluster Trace. For the
first runnable milestone, the default pipeline uses a small Zenodo-hosted
Alibaba 2018 machine-usage sample grouped at 300 seconds.

Google ClusterData 2011 v2 is publicly accessible from the `clusterdata-2011-2`
Google Storage bucket. CPU utilization is read from the `task_usage` table, then
summed per machine and resampled into 5-minute bins. Download a starter Google
shard with:

```powershell
python scripts/download_data.py --source google
python data/preprocess.py --source google --max-rows 2000000
```

The official Alibaba Cluster Trace 2018 release requires a short survey before
download and is too large for an automatic first-run download. The Zenodo sample
is processed from the original Alibaba 2018 files and is enough for C-01 model
pipeline development.
