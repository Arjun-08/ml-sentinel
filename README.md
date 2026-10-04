# ML Sentinel: An AI-Powered Model Reliability

ML Sentinel is a lightweight MLOps portfolio project that demonstrates how a machine-learning team can monitor input data, measure model quality, and investigate performance changes. It generates reproducible synthetic transaction data, trains a fraud classifier, simulates production drift, calculates statistical drift signals, and exports an evidence-based report.

The application works locally without paid APIs, external datasets, a GPU, or an API key. An optional integration can use an Ollama model installed on your machine to explain the computed report.

## Why this project matters

A model that performs well during development can become less reliable when production data changes. Monitoring therefore needs to consider both input distributions and predictive performance. ML Sentinel brings those checks into one interactive dashboard.

## Features

- Reproducible synthetic transaction data; no dataset download is needed.
- Scikit-learn HistGradientBoosting fraud classifier.
- Baseline versus simulated-production evaluation.
- ROC-AUC, PR-AUC (average precision), precision, recall, F1, and confusion matrix.
- Kolmogorov–Smirnov (KS) distribution tests and Population Stability Index (PSI).
- Configurable baseline, feature-drift, and feature-plus-concept-drift scenarios.
- Feature distribution plots, PSI chart, metric comparisons, and confusion matrix.
- Downloadable Markdown investigation report.
- Optional local Ollama report explanation; core features work without it.
- Status messages and reproducible random seeds.

## Architecture

```text
Synthetic transaction generator
              |
              v
      Training data --------------------+
              |                         |
              v                         |
    HistGradientBoosting                |
              |                         |
              v                         |
    Trained fraud classifier            |
              |                         |
      +-------+------------------+      |
      |                          |      |
      v                          v      |
Baseline evaluation       Simulated production
      |                          |
      +-------------+------------+
                    |
          +---------+----------+
          |                    |
          v                    v
    Model metrics        Feature drift
  ROC-AUC, PR-AUC,       KS test, PSI
  precision, recall
          |                    |
          +---------+----------+
                    |
                    v
         Investigation report
                    |
                    v
             Streamlit UI
                    |
             Optional Ollama
```


### Scenarios

| Scenario | What changes | What to investigate |
|---|---|---|
| Baseline | No intentional production shift | Expected reference behavior |
| Feature drift | Transaction input distributions shift | Which features changed and whether performance moved |
| Feature + concept drift | Inputs shift and the label-generating mechanism changes | Whether predictive quality degrades alongside drift |

Because the data is randomly generated, exact values depend on the selected seed and sample size. Use the same settings to reproduce a run.

## Metrics and interpretation

| Metric | What it measures | Why it is included |
|---|---|---|
| ROC-AUC | Ranking quality across thresholds | Broad ranking assessment |
| PR-AUC / average precision | Precision-recall trade-off for the positive class | Useful for imbalanced classification |
| Precision | Fraction of predicted fraud cases that are fraud | Helps reason about false alarms |
| Recall | Fraction of fraud cases detected | Helps reason about missed fraud |
| F1 | Harmonic mean of precision and recall | Compact threshold-specific summary |
| KS statistic and p-value | Difference between two empirical feature distributions | Detects distribution changes |
| PSI | Binned distribution shift | Gives an effect-size-style drift indicator |
| Confusion matrix | True/false positives and negatives | Makes error types visible |

The app uses a fixed probability threshold of 0.5 for transparency. A real deployment should select thresholds on a separate validation set based on operational costs.

## Statistical caveats

- A small KS p-value is evidence of distributional difference, not proof that model quality has declined.
- KS p-values depend on sample size. Inspect the KS statistic, PSI, and plots as well.
- PSI thresholds such as 0.10 or 0.20 are heuristics and should be calibrated for the application.
- The simulated production labels are available immediately. In real fraud systems, labels can be delayed, incomplete, or biased.
- Drift is not causality. Investigate data pipelines, upstream systems, policy changes, and label quality before taking action.
- Synthetic performance does not represent real-world fraud detection performance.

## Optional local LLM with Ollama

The project does not require a language model. To add local natural-language explanation:

1. Install [Ollama](https://ollama.com/) for your operating system.
2. Download a model supported by your machine, for example:
   ```bash
   ollama pull llama3.2
   ```
3. Ensure Ollama is running locally.
4. In the app's **Investigation report** tab, keep the endpoint as `http://localhost:11434`, enter `llama3.2`, and click **Explain report with local model**.

This calls Ollama's local `/api/generate` endpoint. No cloud API key is used. Model names and hardware requirements vary; if this optional step fails, the dashboard and report still work.


## Future improvements

- Support uploaded CSV files with schema validation.
- Add time-window monitoring and delayed-label evaluation.
- Tune thresholds using explicit false-positive and false-negative costs.
- Add configurable alert rules and historical monitoring snapshots.
- Add model calibration, confidence intervals, and model-version comparisons.
- Add automated tests and CI.
- Add privacy-aware logging and production observability.

## Responsible use

This is an educational portfolio project using synthetic data. Do not upload sensitive financial or personal data to a public deployment. Before using the approach in a real system, validate it on authorized data, review fairness and security risks, and establish human-reviewed response procedures.


