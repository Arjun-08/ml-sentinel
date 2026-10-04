from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURES = [
    "amount",
    "account_age_days",
    "transactions_24h",
    "distance_from_home_km",
    "merchant_risk",
]
TARGET = "is_fraud"
OUTPUT_ROOT = Path("outputs")
SCENARIOS = {"Baseline", "Feature drift", "Feature + concept drift"}


def log_step(step: int, total: int, message: str) -> None:
    print(f"\n[{step:02d}/{total:02d}] {message}", flush=True)


def generate_data(n: int, seed: int, scenario: str = "Baseline") -> pd.DataFrame:
    """Generate reproducible synthetic transactions with a consistent target rule."""
    rng = np.random.default_rng(seed)

    log_amount = rng.normal(loc=3.35, scale=0.85, size=n)
    amount = np.exp(log_amount)
    account_age = np.clip(rng.gamma(shape=2.2, scale=220, size=n), 1, 3000)
    tx_24h = np.clip(rng.poisson(lam=2.2, size=n), 0, 25)
    distance = np.clip(rng.exponential(scale=18, size=n), 0, 500)
    merchant_risk = rng.beta(a=2.0, b=5.0, size=n)

    # Production feature drift changes feature distributions, not the code path.
    if scenario in ("Feature drift", "Feature + concept drift"):
        amount *= rng.choice([1.0, 1.8], size=n, p=[0.25, 0.75])
        distance = np.clip(distance * 1.7 + rng.normal(8, 5, n), 0, 700)
        tx_24h = np.clip(tx_24h + rng.poisson(1.5, n), 0, 30)
        merchant_risk = np.clip(merchant_risk * 0.8 + 0.12, 0, 1)

    # Use centered/scaled terms so no single raw feature dominates by accident.
    log_amount_feature = (np.log(np.maximum(amount, 1e-8)) - 3.35) / 0.85
    age_feature = (account_age - 484.0) / 300.0
    tx_feature = (tx_24h - 2.2) / 1.8
    distance_feature = (distance - 18.0) / 25.0
    risk_feature = (merchant_risk - 0.2857) / 0.18

    # Baseline relationship is intentionally learnable but not deterministic.
    z = (
        -3.55
        + 0.65 * log_amount_feature
        - 0.32 * age_feature
        + 0.38 * tx_feature
        + 0.22 * distance_feature
        + 0.70 * risk_feature
    )

    # Concept drift changes the relationship between features and fraud risk.
    if scenario == "Feature + concept drift":
        z = (
            -3.40
            + 0.30 * log_amount_feature
            - 0.10 * age_feature
            + 0.62 * tx_feature
            + 0.12 * distance_feature
            + 1.00 * risk_feature
        )

    probability = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
    labels = rng.binomial(1, probability)

    return pd.DataFrame({
        "amount": amount,
        "account_age_days": account_age,
        "transactions_24h": tx_24h,
        "distance_from_home_km": distance,
        "merchant_risk": merchant_risk,
        TARGET: labels.astype(int),
    })


def predict_probability(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[FEATURES])[:, 1]


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Choose a threshold on validation data only; do not tune on test/production."""
    thresholds = np.linspace(0.05, 0.80, 151)
    scored = [
        (float(f1_score(y_true, probabilities >= threshold, zero_division=0)), float(threshold))
        for threshold in thresholds
    ]
    best_f1, best_threshold = max(scored, key=lambda item: (item[0], -item[1]))
    return best_threshold, best_f1


def calculate_metrics(
    model, frame: pd.DataFrame, name: str, threshold: float
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    y_true = frame[TARGET].to_numpy()
    probabilities = predict_probability(model, frame)
    predictions = (probabilities >= threshold).astype(int)
    row = {
        "Dataset": name,
        "Rows": int(len(frame)),
        "Fraud rate": float(y_true.mean()),
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else float("nan"),
        "PR-AUC": float(average_precision_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    return row, y_true, predictions, probabilities


def calculate_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        ref = reference[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        cur = current[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        ks_stat, p_value = ks_2samp(ref, cur)

        # PSI uses fixed quantile bins learned from the reference distribution.
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            psi = 0.0
        else:
            edges = edges.copy()
            edges[0], edges[-1] = -np.inf, np.inf
            ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
            cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
            ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
            cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        rows.append({
            "Feature": feature,
            "KS statistic": float(ks_stat),
            "KS p-value": float(p_value),
            "PSI": psi,
            "Flagged": bool(p_value < 0.05 or psi >= 0.20),
        })
    return pd.DataFrame(rows).sort_values(["Flagged", "PSI"], ascending=[False, False])


def build_report(
    scenario: str,
    train_rows: int,
    eval_rows: int,
    seed: int,
    threshold: float,
    validation_f1: float,
    metrics: pd.DataFrame,
    drift: pd.DataFrame,
) -> str:
    baseline = metrics.loc[metrics["Dataset"] == "Baseline evaluation"].iloc[0]
    production = metrics.loc[metrics["Dataset"] == "Simulated production"].iloc[0]
    f1_change = float(production["F1"] - baseline["F1"])
    flagged = drift.loc[drift["Flagged"], "Feature"].tolist()

    if f1_change <= -0.05:
        assessment = "PERFORMANCE REGRESSION: production F1 decreased by at least 0.05."
    elif flagged:
        assessment = "DRIFT DETECTED: investigate the flagged features and production behavior."
    else:
        assessment = "NO CONFIGURED DRIFT FLAGS: continue monitoring; this is not proof of stability."

    lines = [
        "# ML Sentinel — Step-by-step experiment report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Production scenario: {scenario}",
        f"- Random seed: {seed}",
        f"- Training rows: {train_rows:,}",
        f"- Evaluation rows per dataset: {eval_rows:,}",
        f"- Validation-selected classification threshold: {threshold:.3f}",
        f"- Validation F1 at selected threshold: {validation_f1:.4f}",
        f"- Overall assessment: **{assessment}**",
        "",
        "## Step 1 — Data generation",
        "A reproducible synthetic transaction dataset was generated locally. No external dataset or paid API was used.",
        f"- Features: {', '.join(FEATURES)}",
        f"- Target: `{TARGET}`",
        "- The training, validation, baseline evaluation, and production evaluation datasets use separate random seeds.",
        "",
        "## Step 2 — Model training",
        "A `HistGradientBoostingClassifier` was trained on the baseline training dataset.",
        "- Configuration: max_iter=160, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=2.0",
        "- The model is a demonstration model trained on synthetic data, not a validated real-world fraud detector.",
        "",
        "## Step 3 — Threshold selection",
        "The probability threshold was selected by maximizing F1 across a threshold grid on a separate baseline validation dataset.",
        "- The baseline evaluation and simulated production datasets were not used to select the threshold.",
        "",
        "## Step 4 — Evaluation metrics",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Step 5 — Baseline versus production",
        f"- ROC-AUC change: {production['ROC-AUC'] - baseline['ROC-AUC']:+.4f}",
        f"- PR-AUC change: {production['PR-AUC'] - baseline['PR-AUC']:+.4f}",
        f"- Precision change: {production['Precision'] - baseline['Precision']:+.4f}",
        f"- Recall change: {production['Recall'] - baseline['Recall']:+.4f}",
        f"- F1 change: {f1_change:+.4f}",
        f"- Baseline fraud rate: {baseline['Fraud rate']:.2%}",
        f"- Production fraud rate: {production['Fraud rate']:.2%}",
        "",
        "## Step 6 — Feature drift analysis",
        "The two-sample Kolmogorov–Smirnov test and Population Stability Index (PSI) were calculated for each feature.",
        drift.to_markdown(index=False, floatfmt=".5f"),
        "",
        "### Features flagged for review",
    ]
    lines.extend([f"- `{feature}`" for feature in flagged] if flagged else ["- No features were flagged by the configured rules."])
    lines.extend([
        "",
        "## Step 7 — Interpretation and limitations",
        f"- Assessment: **{assessment}**",
        "- A KS p-value below 0.05 or PSI of at least 0.20 triggers a feature flag in this demonstration.",
        "- Feature drift does not automatically mean that model performance has degraded.",
        "- Concept drift is represented by changing the synthetic label-generating relationship in the selected scenario.",
        "- Results are synthetic and should not be presented as real-world fraud detection performance.",
        "- In a real deployment, investigate data quality, upstream changes, label delays, and operational costs.",
        "",
        "## Generated artifacts",
        "- `metrics.csv`: evaluation metrics for baseline and simulated production.",
        "- `drift_report.csv`: per-feature KS and PSI results.",
        "- `confusion_matrix_production.png`: production confusion matrix.",
        "- `feature_psi.png`: PSI by feature.",
        "- `run_summary.json`: machine-readable run settings and summary.",
        "- CSV files containing generated training, validation, baseline, and production data.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    print("=" * 72)
    print("ML SENTINEL — REPRODUCIBLE MODEL RELIABILITY PIPELINE")
    print("=" * 72)

    train_rows = int(os.environ.get("ML_SENTINEL_TRAIN_ROWS", "12000"))
    eval_rows = int(os.environ.get("ML_SENTINEL_EVAL_ROWS", "2500"))
    seed = int(os.environ.get("ML_SENTINEL_SEED", "42"))
    scenario = os.environ.get("ML_SENTINEL_SCENARIO", "Feature + concept drift")
    if scenario not in SCENARIOS:
        raise ValueError(f"ML_SENTINEL_SCENARIO must be one of: {sorted(SCENARIOS)}")
    if train_rows < 1000 or eval_rows < 200:
        raise ValueError("Use at least 1,000 training rows and 200 evaluation rows.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    total_steps = 8

    log_step(1, total_steps, "Preparing experiment settings")
    print(f"  Training rows: {train_rows:,}")
    print(f"  Rows per evaluation set: {eval_rows:,}")
    print(f"  Random seed: {seed}")
    print(f"  Production scenario: {scenario}")
    print(f"  Output folder: {out_dir}")

    log_step(2, total_steps, "Generating separate training, validation, and evaluation datasets")
    train = generate_data(train_rows, seed, "Baseline")
    validation = generate_data(eval_rows, seed + 1, "Baseline")
    baseline = generate_data(eval_rows, seed + 2, "Baseline")
    production = generate_data(eval_rows, seed + 3, scenario)
    print(f"  Training fraud rate: {train[TARGET].mean():.2%}")
    print(f"  Validation fraud rate: {validation[TARGET].mean():.2%}")
    print(f"  Baseline evaluation fraud rate: {baseline[TARGET].mean():.2%}")
    print(f"  Production fraud rate: {production[TARGET].mean():.2%}")

    train.to_csv(out_dir / "training_data.csv", index=False)
    validation.to_csv(out_dir / "validation_data.csv", index=False)
    baseline.to_csv(out_dir / "baseline_evaluation_data.csv", index=False)
    production.to_csv(out_dir / "production_evaluation_data.csv", index=False)

    log_step(3, total_steps, "Training HistGradientBoostingClassifier")
    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=seed,
    )
    model.fit(train[FEATURES], train[TARGET])
    print("  Model training completed successfully.")

    log_step(4, total_steps, "Selecting classification threshold on validation data")
    val_probabilities = predict_probability(model, validation)
    threshold, validation_f1 = choose_threshold(validation[TARGET].to_numpy(), val_probabilities)
    print(f"  Selected threshold: {threshold:.3f}")
    print(f"  Validation F1 at selected threshold: {validation_f1:.4f}")

    log_step(5, total_steps, "Evaluating untouched baseline and simulated production data")
    base_row, _, _, _ = calculate_metrics(model, baseline, "Baseline evaluation", threshold)
    prod_row, y_prod, pred_prod, _ = calculate_metrics(model, production, "Simulated production", threshold)
    metrics = pd.DataFrame([base_row, prod_row])
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    log_step(6, total_steps, "Checking feature drift with KS tests and PSI")
    drift = calculate_drift(baseline, production)
    drift.to_csv(out_dir / "drift_report.csv", index=False)
    print(drift.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    log_step(7, total_steps, "Generating charts")
    cm = confusion_matrix(y_prod, pred_prod, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not fraud", "Fraud"]
    ).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("Simulated production confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_production.png", dpi=160)
    plt.close(fig)

    plot_data = drift.sort_values("PSI", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_data["Feature"], plot_data["PSI"])
    ax.axvline(0.10, linestyle="--", label="PSI 0.10 reference")
    ax.axvline(0.20, linestyle="--", label="PSI 0.20 flag threshold")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature drift summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "feature_psi.png", dpi=160)
    plt.close(fig)
    print("  Saved confusion matrix and feature PSI chart.")

    log_step(8, total_steps, "Writing detailed Markdown report and JSON summary")
    report = build_report(
        scenario, train_rows, eval_rows, seed, threshold, validation_f1, metrics, drift
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    summary = {
        "run_id": run_id,
        "scenario": scenario,
        "seed": seed,
        "training_rows": train_rows,
        "evaluation_rows_each": eval_rows,
        "selected_threshold": float(threshold),
        "validation_f1": float(validation_f1),
        "baseline_f1": float(base_row["F1"]),
        "production_f1": float(prod_row["F1"]),
        "f1_change": float(prod_row["F1"] - base_row["F1"]),
        "flagged_features": drift.loc[drift["Flagged"], "Feature"].tolist(),
        "output_directory": str(out_dir),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"  Report saved: {out_dir / 'report.md'}")
    print(f"  Metrics saved: {out_dir / 'metrics.csv'}")
    print(f"  Drift results saved: {out_dir / 'drift_report.csv'}")
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETED")
    print(f"Full report: {out_dir / 'report.md'}")
    print(f"All artifacts: {out_dir.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()