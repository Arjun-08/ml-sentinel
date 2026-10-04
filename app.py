from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ML Sentinel", page_icon="MS", layout="wide")

FEATURES = [
    "amount",
    "account_age_days",
    "transactions_24h",
    "distance_from_home_km",
    "merchant_risk",
]
TARGET = "is_fraud"


def generate_data(n: int, seed: int, scenario: str = "Baseline") -> pd.DataFrame:
    """Create reproducible synthetic transaction data; no external dataset required."""
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=3.4, sigma=0.85, size=n)
    account_age = np.clip(rng.gamma(shape=2.2, scale=220, size=n), 1, 3000)
    tx_24h = np.clip(rng.poisson(lam=2.2, size=n), 0, 25)
    distance = np.clip(rng.exponential(scale=18, size=n), 0, 500)
    merchant_risk = rng.beta(a=2.0, b=5.0, size=n)

    if scenario in ("Feature drift", "Feature + concept drift"):
        amount *= rng.choice([1.0, 1.8], size=n, p=[0.25, 0.75])
        distance = np.clip(distance * 1.7 + rng.normal(8, 5, n), 0, 700)
        tx_24h = np.clip(tx_24h + rng.poisson(1.5, n), 0, 30)
        merchant_risk = np.clip(merchant_risk * 0.8 + 0.12, 0, 1)

    # A latent risk score creates a learnable, imbalanced classification task.
    z = (
        -4.1
        + 0.0065 * amount
        - 0.0010 * account_age
        + 0.19 * tx_24h
        + 0.018 * distance
        + 3.4 * merchant_risk
    )
    if scenario == "Feature + concept drift":
        # Simulate a changed fraud mechanism: old patterns become less predictive.
        z = (
            -3.6
            + 0.0030 * amount
            - 0.00025 * account_age
            + 0.32 * tx_24h
            + 0.009 * distance
            + 4.4 * merchant_risk
        )
    probability = 1 / (1 + np.exp(-np.clip(z, -20, 20)))
    labels = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "amount": amount,
            "account_age_days": account_age,
            "transactions_24h": tx_24h,
            "distance_from_home_km": distance,
            "merchant_risk": merchant_risk,
            "is_fraud": labels.astype(int),
        }
    )


@st.cache_data(show_spinner=False)
def build_experiment(n_train: int, n_eval: int, seed: int, scenario: str):
    train = generate_data(n_train, seed, "Baseline")
    baseline = generate_data(n_eval, seed + 1, "Baseline")
    production = generate_data(n_eval, seed + 2, scenario)
    model = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=20, l2_regularization=1.0,
        random_state=seed
    )
    model.fit(train[FEATURES], train[TARGET])
    return train, baseline, production, model


def metric_row(model, frame: pd.DataFrame, label: str) -> dict:
    y = frame[TARGET].to_numpy()
    proba = model.predict_proba(frame[FEATURES])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "Dataset": label,
        "Fraud rate": float(y.mean()),
        "ROC-AUC": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "PR-AUC": float(average_precision_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def calculate_drift(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        a = reference[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        b = current[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        statistic, p_value = ks_2samp(a, b)
        # PSI based on reference quantile bins. Small epsilon handles empty bins.
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            psi = 0.0
        else:
            edges[0], edges[-1] = -np.inf, np.inf
            ref_counts = np.histogram(a, bins=edges)[0].astype(float)
            cur_counts = np.histogram(b, bins=edges)[0].astype(float)
            ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
            cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        rows.append(
            {
                "Feature": feature,
                "KS statistic": float(statistic),
                "KS p-value": float(p_value),
                "PSI": psi,
                "Flagged": bool(p_value < 0.05 or psi >= 0.2),
            }
        )
    return pd.DataFrame(rows).sort_values(["Flagged", "PSI"], ascending=[False, False])


def make_report(scenario: str, metrics: pd.DataFrame, drift: pd.DataFrame) -> str:
    base = metrics.loc[metrics["Dataset"] == "Baseline evaluation"].iloc[0]
    prod = metrics.loc[metrics["Dataset"] == "Simulated production"].iloc[0]
    delta = prod["F1"] - base["F1"]
    flagged = drift.loc[drift["Flagged"], "Feature"].tolist()
    if delta <= -0.05:
        status = "HIGH ATTENTION: F1 decreased by at least 0.05 on simulated production data."
    elif flagged:
        status = "INVESTIGATE: feature-distribution drift was detected; performance impact should be monitored."
    else:
        status = "NO MAJOR SIGNAL: no feature crossed the configured drift flags in this run."
    lines = [
        "# ML Sentinel investigation report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Scenario: {scenario}",
        f"- Assessment: **{status}**",
        "",
        "## Performance comparison",
        f"- Baseline ROC-AUC: {base['ROC-AUC']:.3f}",
        f"- Production ROC-AUC: {prod['ROC-AUC']:.3f}",
        f"- Baseline PR-AUC: {base['PR-AUC']:.3f}",
        f"- Production PR-AUC: {prod['PR-AUC']:.3f}",
        f"- Baseline F1: {base['F1']:.3f}",
        f"- Production F1: {prod['F1']:.3f}",
        f"- F1 change (production - baseline): {delta:+.3f}",
        f"- Baseline fraud rate: {base['Fraud rate']:.2%}",
        f"- Production fraud rate: {prod['Fraud rate']:.2%}",
        "",
        "## Drift findings",
    ]
    if flagged:
        lines.extend([f"- Flagged feature: `{name}`" for name in flagged])
    else:
        lines.append("- No features were flagged by the KS/PSI rules.")
    lines += [
        "",
        "## Interpretation and limitations",
        "- KS p-values are sensitive to sample size; interpret them alongside effect size (KS statistic) and PSI.",
        "- PSI >= 0.20 is used here as a heuristic, not a universal production policy.",
        "- Production labels are available because this is a controlled simulation. Real systems often have delayed labels.",
        "- This experiment does not prove causality. Investigate upstream data pipelines and validate with real, approved data before acting.",
    ]
    return "\n".join(lines)


def ask_ollama(report: str, model_name: str, endpoint: str) -> str:
    prompt = (
        "You are an ML reliability analyst. Explain the following report in concise, cautious language. "
        "Use only evidence in the report. Separate observed facts from hypotheses and suggest three checks.\n\n"
        + report
    )
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("response", "The local model returned no text.")


st.title("ML Sentinel")
st.subheader("Model reliability, data drift, and evidence-based investigation")
st.write(
    "A reproducible ML monitoring lab. Generate a baseline, simulate production drift, "
    "measure performance, and export an investigation report. All data is synthetic."
)

with st.sidebar:
    st.header("Experiment settings")
    n_train = st.select_slider("Training rows", options=[5000, 10000, 20000, 30000], value=10000)
    n_eval = st.select_slider("Rows per evaluation set", options=[1000, 2000, 4000, 6000], value=2000)
    seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)
    scenario = st.selectbox(
        "Production scenario",
        ["Baseline", "Feature drift", "Feature + concept drift"],
        index=2,
        help="Feature drift changes input distributions. Concept drift also changes the label-generating mechanism.",
    )
    run = st.button("Run experiment", type="primary", use_container_width=True)
    st.caption("First run trains the model; subsequent identical runs use cached results.")

if "experiment" not in st.session_state:
    st.session_state.experiment = None
if run or st.session_state.experiment is None:
    with st.spinner("Generating data, training model, and calculating drift..."):
        st.write("Status: generating reproducible synthetic datasets...")
        experiment = build_experiment(int(n_train), int(n_eval), int(seed), scenario)
        st.write("Status: model training completed.")
        st.write("Status: calculating metrics and feature drift...")
        st.session_state.experiment = (int(n_train), int(n_eval), int(seed), scenario, experiment)

saved_n_train, saved_n_eval, saved_seed, saved_scenario, result = st.session_state.experiment
train, baseline, production, model = result
if (saved_n_train, saved_n_eval, saved_seed, saved_scenario) != (int(n_train), int(n_eval), int(seed), scenario):
    st.info("Settings changed. Click **Run experiment** to run the selected configuration.")

metrics = pd.DataFrame([
    metric_row(model, baseline, "Baseline evaluation"),
    metric_row(model, production, "Simulated production"),
])
drift = calculate_drift(baseline, production)
report = make_report(saved_scenario, metrics, drift)

base_m = metrics.iloc[0]
prod_m = metrics.iloc[1]
delta_f1 = prod_m["F1"] - base_m["F1"]
flag_count = int(drift["Flagged"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Production ROC-AUC", f"{prod_m['ROC-AUC']:.3f}", f"{prod_m['ROC-AUC']-base_m['ROC-AUC']:+.3f} vs baseline")
c2.metric("Production PR-AUC", f"{prod_m['PR-AUC']:.3f}", f"{prod_m['PR-AUC']-base_m['PR-AUC']:+.3f} vs baseline")
c3.metric("Production F1", f"{prod_m['F1']:.3f}", f"{delta_f1:+.3f} vs baseline")
c4.metric("Features flagged", f"{flag_count}/{len(FEATURES)}")

if delta_f1 <= -0.05:
    st.error("Performance degradation signal: production F1 is at least 0.05 below baseline in this simulation.")
elif flag_count:
    st.warning("Distribution drift detected. Check the flagged features and compare it with performance changes.")
else:
    st.success("No feature drift flags in this run. This is not proof that the model is safe in production.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Feature drift", "Performance", "Investigation report"])

with tab1:
    st.markdown("### Experiment summary")
    st.dataframe(metrics.style.format({
        "Fraud rate": "{:.2%}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
        "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"
    }), use_container_width=True, hide_index=True)
    st.markdown("### Dataset snapshot")
    snap = pd.DataFrame({
        "Dataset": ["Training", "Baseline evaluation", "Simulated production"],
        "Rows": [len(train), len(baseline), len(production)],
        "Fraud cases": [int(train[TARGET].sum()), int(baseline[TARGET].sum()), int(production[TARGET].sum())],
        "Fraud rate": [train[TARGET].mean(), baseline[TARGET].mean(), production[TARGET].mean()],
    })
    st.dataframe(snap.style.format({"Fraud rate": "{:.2%}"}), use_container_width=True, hide_index=True)
    st.caption("Synthetic results are for demonstrating the monitoring workflow, not estimating real fraud performance.")

with tab2:
    st.markdown("### Statistical drift checks")
    st.write("KS test compares empirical distributions. PSI summarizes changes across reference quantile bins.")
    st.dataframe(drift.style.format({"KS statistic": "{:.3f}", "KS p-value": "{:.3g}", "PSI": "{:.3f}"}), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_drift = drift.sort_values("PSI", ascending=True)
    ax.barh(plot_drift["Feature"], plot_drift["PSI"])
    ax.axvline(0.1, linestyle="--", label="PSI 0.10 reference")
    ax.axvline(0.2, linestyle="--", label="PSI 0.20 flag")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature drift by PSI")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    chosen_feature = st.selectbox("Inspect feature distribution", FEATURES, key="feature_choice")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(baseline[chosen_feature], bins=35, alpha=0.55, density=True, label="Baseline")
    ax.hist(production[chosen_feature], bins=35, alpha=0.55, density=True, label="Production")
    ax.set_title(f"Distribution comparison: {chosen_feature}")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.markdown("### Model quality")
    plot_metrics = metrics.set_index("Dataset")[["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]].T
    st.bar_chart(plot_metrics)
    y_true = production[TARGET].to_numpy()
    y_pred = (model.predict_proba(production[FEATURES])[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Predicted normal", "Predicted fraud"])
    ax.set_yticks([0, 1], labels=["Actual normal", "Actual fraud"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title("Production confusion matrix")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The 0.5 threshold is fixed for transparency. Threshold tuning should use a separate validation set and an explicit business cost.")

with tab4:
    st.markdown(report)
    st.download_button(
        "Download investigation report (.md)",
        data=report.encode("utf-8"),
        file_name="ml_sentinel_investigation_report.md",
        mime="text/markdown",
    )
    st.markdown("### Optional local LLM explanation")
    st.write("If Ollama is installed and running locally, you can ask a local model to explain this report. No paid API is required.")
    ollama_endpoint = st.text_input("Ollama endpoint", value="http://localhost:11434")
    ollama_model = st.text_input("Installed Ollama model", value="llama3.2")
    if st.button("Explain report with local model"):
        try:
            with st.spinner("Contacting local Ollama server..."):
                explanation = ask_ollama(report, ollama_model, ollama_endpoint)
            st.markdown(explanation)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            st.error(f"Could not reach the local model: {exc}. The metrics and rule-based report are still available.")

with st.expander("Methodology and caveats"):
    st.markdown(
        """
        - **Data generation:** reproducible synthetic transaction records with an intentionally learnable fraud signal.
        - **Feature drift:** shifts input feature distributions.
        - **Concept drift:** changes the mechanism used to generate labels, representing a change in the relationship between features and outcome.
        - **KS test:** flags distribution differences; p-values can become small for modest changes with large samples.
        - **PSI:** uses decile-like bins from the baseline data. Thresholds are practical heuristics, not universal standards.
        - **Limitations:** synthetic data is not a substitute for validation on authorized, representative production data. This dashboard does not establish causality or automatically retrain/deploy a model.
        """
    )

st.divider()
st.caption("ML Sentinel | Synthetic demonstration | Built with Python, scikit-learn, SciPy, Matplotlib, and Streamlit")
