# DeepSched Stage 1 Report

Stage 1 is the prototype-development phase of DeepSched. Its goal is to make
the complete research pipeline executable end to end on a compact dataset before
scaling the experiments to larger Alibaba and Google traces in future work.

Paper title:

**DeepSched: A Confidence-Aware Adaptive CPU Scheduling Framework Using Deep
Workload Prediction**

## Stage 1 Scope

Stage 1 includes:

- C-01 data download and preprocessing
- C-02 LSTM workload prediction baseline
- C-03 CNN-LSTM-Attention workload prediction model
- C-04 WPM ablation studies
- C-04.5 forecast quality analysis
- C-05 Gymnasium CPU scheduling environment
- C-06 classical schedulers and confidence-aware adaptive advisor
- C-07 PPO scheduling agent
- C-08 figure builder
- C-09 Stage 1 result summary and conclusion

Stage 1 does not claim final paper-level generalization. The current dataset is
a compact Alibaba-derived sample intended for development and pipeline
validation.

## Results So Far

### Workload Prediction

The CNN-LSTM-Attention model is the strongest Stage 1 forecasting architecture.

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| LSTM only | 0.0778 | 0.1022 | 20.09% |
| CNN-LSTM | 0.0751 | 0.0991 | 19.31% |
| LSTM-Attention | 0.0772 | 0.1016 | 19.81% |
| CNN-LSTM-Attention | 0.0744 | 0.0977 | 19.12% |

The ablation results support the intended WPM design: CNN feature extraction
helps, attention alone is not enough, and the combined CNN-LSTM-Attention model
gives the best Stage 1 RMSE.

### Forecast Horizon

| Horizon | MAE | RMSE | MAPE |
| ---: | ---: | ---: | ---: |
| H=1 | 0.0620 | 0.0840 | 15.32% |
| H=5 | 0.0698 | 0.0923 | 18.27% |
| H=10 | 0.0759 | 0.1005 | 18.89% |

Shorter horizons are easier to predict, which is expected for CPU-utilization
forecasting.

### Forecast Quality Analysis

Forecast quality analysis shows that aggregate forecasting metrics are not
sufficient. Burst and peak detection are currently weak at the selected
thresholds.

| Model | Event | Threshold | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM baseline | Burst | 0.75 | 0.00 | 0.00 | 0.00 |
| LSTM baseline | Peak | 0.90 | 0.00 | 0.00 | 0.00 |
| CNN-LSTM-Attention | Burst | 0.75 | 0.00 | 0.00 | 0.00 |
| CNN-LSTM-Attention | Peak | 0.90 | 0.00 | 0.00 | 0.00 |

This is a useful Stage 1 finding: the model can reduce average forecast error,
but the current dataset and loss formulation are not enough for robust
burst/peak prediction.

### Confidence-Aware Adaptive Scheduling

The Confidence_Rule_WPM scheduler improves the deterministic Stage 1 scheduling
simulation. It does not blindly trust workload forecasts. Instead, it estimates
forecast confidence from horizon stability and uses predictions only when the
confidence score is high enough. If confidence is low, it falls back to FCFS.

| Scheduler | AWT | ATT | Throughput |
| --- | ---: | ---: | ---: |
| FCFS | 18.47 | 34.43 | 0.0333 |
| SJF | 13.51 | 29.47 | 0.0333 |
| RR-20 | 19.19 | 35.15 | 0.0333 |
| Confidence_Rule_WPM | 11.54 | 27.50 | 0.0333 |

The mean confidence score is 0.8022 and the advisor trusts forecasts for 93.41%
of scheduling decisions. This is currently the strongest interpretable
scheduling result.

### PPO Scheduling

PPO trains successfully in the Gymnasium scheduling environment, but the
forecast-aware PPO agent is not yet stronger than PPO without forecasts.

| Scheduler | AWT | ATT | Mean Reward |
| --- | ---: | ---: | ---: |
| Random | 155.10 | 172.01 | -0.4424 |
| FCFS policy | 153.45 | 170.17 | -0.4407 |
| SJF policy | 84.43 | 101.23 | -0.6650 |
| Confidence_Rule_WPM | 153.45 | 170.17 | -0.4407 |
| PPO no forecast | 75.83 | 92.69 | -0.2973 |
| PPO DeepSched | 89.06 | 105.63 | -0.3321 |

In the Gymnasium environment, Confidence_Rule_WPM currently falls back to FCFS
for most decisions because early episode forecast features are not informative.
This is an implementation limitation to address in future work. PPO is useful as
a working RL baseline, but reward/state tuning is required before making a
stronger PPO contribution.

## Stage 1 Conclusion

Stage 1 successfully implements the full DeepSched pipeline from data
preprocessing to forecasting, scheduling simulation, rule-based scheduling, PPO
training, and figure generation. The results are meaningful for development:

- The WPM architecture is justified by ablation results.
- Forecast horizon behavior follows the expected trend.
- The confidence-aware adaptive advisor improves scheduling metrics in the
  deterministic simulator.
- PPO is operational but not yet the main contribution.

The project is not yet ready for final research-paper claims. The current
results are best described as a validated prototype and Stage 1 feasibility
study.

## Future Work

Future work will rerun the full pipeline on stronger datasets and with improved
evaluation:

- expand Alibaba trace coverage beyond the compact sample
- add Google Cluster Trace experiments
- regenerate processed tensors for each dataset
- retrain LSTM, CNN-LSTM, LSTM-Attention, and CNN-LSTM-Attention models
- rerun horizon and burst/peak forecast-quality analysis
- calibrate confidence thresholds using validation residuals
- compare confidence estimates from horizon stability, validation residuals,
  ensembles, MC dropout, and conformal intervals
- improve PPO state representation and reward design
- evaluate PPO with and without forecasts over more seeds and longer training
- compare results across datasets to test generalization
- prepare final paper tables, figures, and conclusion

The future-work goal is to turn the Stage 1 prototype into a paper-ready
empirical study.
