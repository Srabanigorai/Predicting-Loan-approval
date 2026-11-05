import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score
)

# ── PAGE CONFIG ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1120 0%, #111827 100%);
    color: #e2e8f0;
}

header[data-testid="stHeader"] { background: transparent; }

.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #f5c842 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 0.5rem 0 0.2rem;
    line-height: 1.2;
}
.main-sub {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: linear-gradient(145deg, #1a2540, #0f1a2e);
    border: 1px solid #243057;
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-blue::after   { background: #4f8ef7; }
.kpi-green::after  { background: #34d399; }
.kpi-gold::after   { background: #f5c842; }
.kpi-red::after    { background: #f87171; }

.kpi-icon  { font-size: 1.8rem; margin-bottom: 0.3rem; }
.kpi-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }
.kpi-value { font-size: 2rem; font-weight: 800; margin-top: 0.2rem; }
.kpi-blue  .kpi-value { color: #93c5fd; }
.kpi-green .kpi-value { color: #6ee7b7; }
.kpi-gold  .kpi-value { color: #fde68a; }
.kpi-red   .kpi-value { color: #fca5a5; }

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #4f8ef7;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

.metric-bar-bg {
    background: #1e2d4a;
    border-radius: 99px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 99px;
}

.cm-grid {
    display: grid;
    grid-template-columns: 120px 1fr 1fr;
    grid-template-rows: 40px 1fr 1fr;
    gap: 6px;
    max-width: 360px;
    margin: 0 auto;
}
.cm-header { display: flex; align-items: center; justify-content: center;
             font-size: 0.75rem; font-weight: 600; color: #64748b; }
.cm-cell {
    border-radius: 10px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 1rem;
    font-weight: 700;
}
.cm-tn { background: #1a3a6e44; border: 1px solid #34d39966; }
.cm-tn .cm-val { color: #6ee7b7; font-size: 1.6rem; }
.cm-fp { background: #7c3a1044; border: 1px solid #f8717166; }
.cm-fp .cm-val { color: #fca5a5; font-size: 1.6rem; }
.cm-fn { background: #7c3a1044; border: 1px solid #f8717166; }
.cm-fn .cm-val { color: #fca5a5; font-size: 1.6rem; }
.cm-tp { background: #1a3a6e44; border: 1px solid #34d39966; }
.cm-tp .cm-val { color: #6ee7b7; font-size: 1.6rem; }
.cm-label { font-size: 0.65rem; color: #64748b; margin-top: 0.2rem; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4f8ef7, #2563eb);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 2rem;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s;
    box-shadow: 0 4px 20px #4f8ef740;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px #4f8ef760;
}

section[data-testid="stSidebar"] {
    background: #0f1729;
    border-right: 1px solid #243057;
}

.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── LOAD AND PROCESS DATA (same logic as loan_approval_spark.py) ─────────
@st.cache_data
def load_and_process():
    """Replicate the exact same pipeline as loan_approval_spark.py using pandas + sklearn."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "loan_approval_dataset.csv")
    df = pd.read_csv(csv_path)

    # Clean column names and string values (same as Spark script)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Drop loan_id
    df = df.drop(columns=["loan_id"], errors="ignore")

    # Define columns (same as Spark script)
    numeric_cols = [
        "no_of_dependents", "income_annum", "loan_amount",
        "loan_term", "cibil_score", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value",
        "bank_asset_value"
    ]
    categorical_cols = ["education", "self_employed"]

    # Encode categorical columns
    le_edu = LabelEncoder()
    le_emp = LabelEncoder()
    df["education_index"] = le_edu.fit_transform(df["education"])
    df["self_employed_index"] = le_emp.fit_transform(df["self_employed"])

    # Encode target: Approved=1, Rejected=0
    df["label"] = (df["loan_status"] == "Approved").astype(int)

    # Split 80/20 with seed=42 (same as Spark script)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Impute missing numeric values with median (same as Spark script)
    imputer = SimpleImputer(strategy="median")
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

    # Standardize CIBIL score (same as Spark script Step 4)
    scaler = StandardScaler()
    train_df["credit_scaled"] = scaler.fit_transform(train_df[["cibil_score"]])
    test_df["credit_scaled"] = scaler.transform(test_df[["cibil_score"]])

    # Assemble features (same as Spark script Step 6)
    feature_cols = (
        [c for c in numeric_cols if c != "cibil_score"]
        + ["credit_scaled"]
        + ["education_index", "self_employed_index"]
    )

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # Train Decision Tree (maxDepth=5, seed=42 — same as Spark script)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    # Predict
    y_pred = dt.predict(X_test)

    # Metrics
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = round(precision_score(y_test, y_pred) * 100, 2)
    rec = round(recall_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)
    spec = round(tn / (tn + fp) * 100, 2) if (tn + fp) > 0 else 0

    results = {
        "accuracy": acc, "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "precision": prec, "recall": rec,
        "f1": f1, "specificity": spec,
        "total_records": len(df),
        "feature_cols": feature_cols,
        "feature_importances": dict(zip(feature_cols, dt.feature_importances_)),
    }

    return df, results


# ── LOAD DATA ─────────────────────────────────────────────────────────────
df, results = load_and_process()


# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Loan Approval")
    st.markdown("**Big Data Analytics Dashboard**")
    st.markdown("---")
    st.markdown("### 📌 Project Info")
    st.markdown("""
- **Algorithm:** Decision Tree  
- **Framework:** Apache Spark (PySpark)  
- **Language:** Python 3  
- **Split:** 80% Train / 20% Test  
- **Max Depth:** 5  
- **Seed:** 42  
""")
    st.markdown("---")
    st.markdown("### 🗂️ Source")
    st.code("loan_approval_spark.py", language="text")
    st.markdown("Dashboard replicates the **exact same pipeline** as the PySpark script.")
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown(f"**{results['total_records']:,}** records · **12** features")
    st.markdown("[Kaggle Dataset ↗](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)")


# ── HEADER ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ Loan Approval Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">Apache Spark · Decision Tree · Real-time ML Dashboard</div>', unsafe_allow_html=True)


# ── KPI CARDS ────────────────────────────────────────────────────────────
approved_count = int((df["loan_status"] == "Approved").sum())
rejected_count = int((df["loan_status"] == "Rejected").sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-card kpi-blue">
        <div class="kpi-icon">🗄️</div>
        <div class="kpi-label">Total Records</div>
        <div class="kpi-value kpi-blue">{results['total_records']:,}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card kpi-green">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value" style="color:#6ee7b7;">{results['accuracy']}%</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card kpi-gold">
        <div class="kpi-icon">✅</div>
        <div class="kpi-label">Approved</div>
        <div class="kpi-value kpi-gold">{approved_count:,}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-card kpi-red">
        <div class="kpi-icon">❌</div>
        <div class="kpi-label">Rejected</div>
        <div class="kpi-value kpi-red">{rejected_count:,}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


# ── TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Results", "📈 Visualizations", "🔍 Dataset Explorer", "📋 Pipeline"]
)


# ────────────────────────────────────────────────────
# TAB 1 — MODEL RESULTS
# ────────────────────────────────────────────────────
with tab1:
    r = results
    tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]

    # Accuracy bar
    st.markdown('<div class="section-title">Model Accuracy</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])

    with col_a:
        pct = r["accuracy"]
        st.markdown(f"""
        <div style="font-size:3.5rem; font-weight:800; color:#6ee7b7; line-height:1;">{pct}%</div>
        <div class="metric-bar-bg" style="margin-top:0.8rem;">
          <div class="metric-bar-fill" style="width:{pct}%; background:linear-gradient(90deg,#34d399,#10b981);"></div>
        </div>
        <div style="color:#64748b; font-size:0.82rem; margin-top:0.4rem;">
          DecisionTreeClassifier · 20% test split · seed=42 · maxDepth=5
        </div>""", unsafe_allow_html=True)

    with col_b:
        total_test = tn + fp + fn + tp
        correct = tn + tp
        st.metric("Correct Predictions", f"{correct:,}")
        st.metric("Incorrect Predictions", f"{total_test - correct:,}")
        st.metric("Test Set Size", f"{total_test:,}")

    # Confusion Matrix
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_col, metrics_col = st.columns([1, 1])

    with cm_col:
        st.markdown(f"""
        <div class="cm-grid">
          <div></div>
          <div class="cm-header">Pred: Rejected</div>
          <div class="cm-header">Pred: Approved</div>

          <div class="cm-header" style="justify-content:flex-end; padding-right:8px;">Actual<br>Rejected</div>
          <div class="cm-cell cm-tn">
            <div class="cm-val">{tn}</div>
            <div class="cm-label">True Negative</div>
          </div>
          <div class="cm-cell cm-fp">
            <div class="cm-val">{fp}</div>
            <div class="cm-label">False Positive</div>
          </div>

          <div class="cm-header" style="justify-content:flex-end; padding-right:8px;">Actual<br>Approved</div>
          <div class="cm-cell cm-fn">
            <div class="cm-val">{fn}</div>
            <div class="cm-label">False Negative</div>
          </div>
          <div class="cm-cell cm-tp">
            <div class="cm-val">{tp}</div>
            <div class="cm-label">True Positive</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with metrics_col:
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score", "Specificity"],
            "Value %": [r["accuracy"], r["precision"], r["recall"], r["f1"], r["specificity"]]
        })
        st.dataframe(
            metrics_df.style.format({"Value %": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True
        )

        for name, val, color in [
            ("Precision", r["precision"], "#93c5fd"),
            ("Recall", r["recall"], "#6ee7b7"),
            ("F1 Score", r["f1"], "#fde68a"),
            ("Specificity", r["specificity"], "#c4b5fd"),
        ]:
            st.markdown(f"""
            <div style="margin-top:0.5rem;">
              <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
                <span style="color:#94a3b8;">{name}</span>
                <span style="color:{color};font-weight:700;">{val}%</span>
              </div>
              <div class="metric-bar-bg">
                <div class="metric-bar-fill" style="width:{val}%;background:{color};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    try:
        import plotly.express as px

        imp = results["feature_importances"]
        imp_df = pd.DataFrame({
            "Feature": list(imp.keys()),
            "Importance": [round(v * 100, 2) for v in imp.values()]
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#1e3a5f", "#4f8ef7", "#34d399"],
            text=imp_df["Importance"].apply(lambda x: f"{x:.1f}%")
        )
        fig_imp.update_layout(
            paper_bgcolor="#1a2540", plot_bgcolor="#1a2540",
            font_color="#e2e8f0",
            showlegend=False, coloraxis_showscale=False,
            xaxis_gridcolor="#243057", yaxis_gridcolor="#243057",
            height=400, margin=dict(l=0, r=40, t=10, b=0)
        )
        fig_imp.update_traces(textposition="outside")
        st.plotly_chart(fig_imp, use_container_width=True)
    except ImportError:
        st.info("Install plotly for interactive charts: `pip install plotly`")


# ────────────────────────────────────────────────────
# TAB 2 — VISUALIZATIONS
# ────────────────────────────────────────────────────
with tab2:
    # ── Matplotlib Charts (same as loan_approval_spark.py output) ─────
    r = results
    tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]

    st.markdown('<div class="section-title">Confusion Matrix Heatmap</div>', unsafe_allow_html=True)
    spark_1, spark_2 = st.columns(2)

    with spark_1:
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("#1a2540")
        ax_cm.set_facecolor("#1a2540")
        cm_data = [[tn, fp], [fn, tp]]
        ax_cm.imshow(cm_data, cmap="Blues")
        for i, row in enumerate(cm_data):
            for j, value in enumerate(row):
                ax_cm.text(j, i, str(value), ha="center", va="center",
                           fontsize=18, fontweight="bold", color="white")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_xticklabels(["Rejected", "Approved"], color="#94a3b8")
        ax_cm.set_yticks([0, 1])
        ax_cm.set_yticklabels(["Rejected", "Approved"], color="#94a3b8")
        ax_cm.set_xlabel("Predicted", color="#e2e8f0", fontsize=12)
        ax_cm.set_ylabel("Actual", color="#e2e8f0", fontsize=12)
        ax_cm.set_title("Loan Approval Confusion Matrix", color="#e2e8f0", fontsize=13, pad=12)
        ax_cm.tick_params(colors="#94a3b8")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with spark_2:
        st.markdown('<div class="section-title">Approval Rate by Income Group</div>', unsafe_allow_html=True)
        income_df_mpl = df.copy()
        income_df_mpl["income_group"] = pd.cut(
            income_df_mpl["income_annum"],
            bins=[0, 2_500_000, 5_000_000, float("inf")],
            labels=["Low Income", "Middle Income", "High Income"]
        )
        approval_mpl = (
            income_df_mpl.groupby("income_group", observed=True)
            .apply(lambda x: (x["loan_status"] == "Approved").mean() * 100)
            .reset_index(name="approval_rate")
        )
        fig_inc, ax_inc = plt.subplots(figsize=(7, 5))
        fig_inc.patch.set_facecolor("#1a2540")
        ax_inc.set_facecolor("#1a2540")
        bars = ax_inc.bar(
            approval_mpl["income_group"],
            approval_mpl["approval_rate"],
            color=["#f87171", "#fbbf24", "#34d399"],
            edgecolor="none",
            width=0.55
        )
        for bar in bars:
            h = bar.get_height()
            ax_inc.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                        f"{h:.1f}%", ha="center", va="bottom",
                        color="#e2e8f0", fontweight="bold", fontsize=12)
        ax_inc.set_xlabel("Income Group", color="#e2e8f0", fontsize=12)
        ax_inc.set_ylabel("Approval Rate (%)", color="#e2e8f0", fontsize=12)
        ax_inc.set_title("Loan Approval Rate by Income Group", color="#e2e8f0", fontsize=13, pad=12)
        ax_inc.set_ylim(0, 100)
        ax_inc.tick_params(colors="#94a3b8")
        ax_inc.spines["top"].set_visible(False)
        ax_inc.spines["right"].set_visible(False)
        ax_inc.spines["left"].set_color("#243057")
        ax_inc.spines["bottom"].set_color("#243057")
        fig_inc.tight_layout()
        st.pyplot(fig_inc)
        plt.close(fig_inc)

    st.markdown("---")

    try:
        import plotly.express as px

        st.markdown('<div class="section-title">Interactive Visualizations</div>', unsafe_allow_html=True)

        d1, d2 = st.columns(2)

        with d1:
            counts = df["loan_status"].value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            fig = px.pie(
                counts, names="Status", values="Count",
                title="Approved vs Rejected Split",
                color="Status",
                color_discrete_map={"Approved": "#34d399", "Rejected": "#f87171"},
                hole=0.5
            )
            fig.update_layout(
                paper_bgcolor="#1a2540", plot_bgcolor="#1a2540",
                font_color="#e2e8f0", title_font_size=14,
                legend=dict(font_color="#94a3b8")
            )
            st.plotly_chart(fig, use_container_width=True)

        with d2:
            income_df = df.copy()
            income_df["income_group"] = pd.cut(
                income_df["income_annum"],
                bins=[0, 2_500_000, 5_000_000, float("inf")],
                labels=["Low Income", "Middle Income", "High Income"]
            )
            approval_rate = (
                income_df.groupby("income_group", observed=True)
                .apply(lambda x: (x["loan_status"] == "Approved").mean() * 100)
                .reset_index(name="approval_rate")
            )
            fig2 = px.bar(
                approval_rate, x="income_group", y="approval_rate",
                title="Approval Rate by Income Group (%)",
                color="income_group",
                color_discrete_map={
                    "Low Income": "#f87171",
                    "Middle Income": "#fbbf24",
                    "High Income": "#34d399"
                },
                text=approval_rate["approval_rate"].round(1).astype(str) + "%"
            )
            fig2.update_layout(
                paper_bgcolor="#1a2540", plot_bgcolor="#1a2540",
                font_color="#e2e8f0", title_font_size=14,
                showlegend=False, yaxis_range=[0, 100],
                yaxis_gridcolor="#243057"
            )
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

        # CIBIL Score Distribution
        st.markdown('<div class="section-title">CIBIL Score Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(
            df, x="cibil_score", color="loan_status",
            nbins=40, barmode="overlay",
            title="CIBIL Score Distribution by Loan Status",
            color_discrete_map={"Approved": "#34d399", "Rejected": "#f87171"},
            opacity=0.8
        )
        fig3.update_layout(
            paper_bgcolor="#1a2540", plot_bgcolor="#1a2540",
            font_color="#e2e8f0", title_font_size=14,
            xaxis_gridcolor="#243057", yaxis_gridcolor="#243057",
            legend=dict(font_color="#94a3b8")
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Loan Amount vs Income
        st.markdown('<div class="section-title">Loan Amount vs Annual Income</div>', unsafe_allow_html=True)
        fig4 = px.scatter(
            df, x="income_annum", y="loan_amount",
            color="loan_status",
            color_discrete_map={"Approved": "#34d399", "Rejected": "#f87171"},
            opacity=0.5,
            title="Loan Amount vs Annual Income by Status",
            labels={"income_annum": "Annual Income (₹)", "loan_amount": "Loan Amount (₹)"}
        )
        fig4.update_layout(
            paper_bgcolor="#1a2540", plot_bgcolor="#1a2540",
            font_color="#e2e8f0", title_font_size=14,
            xaxis_gridcolor="#243057", yaxis_gridcolor="#243057",
            legend=dict(font_color="#94a3b8")
        )
        st.plotly_chart(fig4, use_container_width=True)

    except ImportError:
        st.warning("Install plotly for interactive charts: `pip install plotly`")


# ────────────────────────────────────────────────────
# TAB 3 — DATASET EXPLORER
# ────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        edu_filter = st.selectbox("Education", ["All"] + sorted(df["education"].unique().tolist()))
    with fc2:
        emp_filter = st.selectbox("Self Employed", ["All"] + sorted(df["self_employed"].unique().tolist()))
    with fc3:
        status_filter = st.selectbox("Loan Status", ["All", "Approved", "Rejected"])

    filtered = df.copy()
    if edu_filter != "All":
        filtered = filtered[filtered["education"] == edu_filter]
    if emp_filter != "All":
        filtered = filtered[filtered["self_employed"] == emp_filter]
    if status_filter != "All":
        filtered = filtered[filtered["loan_status"] == status_filter]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered, use_container_width=True, height=400)

    st.markdown('<div class="section-title">Column Statistics</div>', unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include="number")
    st.dataframe(
        numeric_df.describe().round(2).T.rename(columns={"50%": "median"}),
        use_container_width=True
    )


# ────────────────────────────────────────────────────
# TAB 4 — PIPELINE
# ────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">PySpark ML Pipeline (from loan_approval_spark.py)</div>',
                unsafe_allow_html=True)

    steps = [
        ("1", "SparkSession", "Creates Spark session in local[*] mode (all CPU cores)", "#4f8ef7"),
        ("2", "Read CSV", "Loads dataset, strips whitespace, drops loan_id column", "#818cf8"),
        ("3", "Imputer", "Fills missing numerical values using median strategy across 9 columns", "#a78bfa"),
        ("4", "StandardScaler", "Normalizes CIBIL score to mean=0, std=1 via z-score scaling", "#c084fc"),
        ("5", "StringIndexer", "Encodes education & self_employed to numeric; loan_status → label (Approved=1)", "#e879f9"),
        ("6", "VectorAssembler", "Merges all 11 feature columns into a single features vector", "#f472b6"),
        ("7", "DecisionTreeClassifier", "Trains Decision Tree (maxDepth=5, seed=42) on 80% training data", "#fb7185"),
        ("8", "Evaluate", "Computes accuracy using MulticlassClassificationEvaluator on 20% test set", "#f97316"),
        ("9", "Confusion Matrix", "Counts TP/TN/FP/FN using PySpark agg + F.when; saves heatmap PNG", "#eab308"),
        ("10", "Income Analysis", "Groups by income tier, computes approval rate, saves bar chart PNG", "#84cc16"),
    ]

    for num, name, desc, color in steps:
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:1rem;
                    background:#1a2540; border:1px solid #243057;
                    border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.6rem;">
          <div style="min-width:36px; height:36px; border-radius:10px; background:{color}22;
                      border:1px solid {color}66; color:{color}; font-weight:800;
                      font-size:0.9rem; display:flex; align-items:center; justify-content:center;">
            {num}
          </div>
          <div>
            <div style="font-weight:700; color:#e2e8f0; font-size:0.95rem;">{name}</div>
            <div style="color:#64748b; font-size:0.83rem; margin-top:0.15rem;">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Pipeline Code (from loan_approval_spark.py)</div>',
                unsafe_allow_html=True)
    st.code("""Pipeline(stages=[
    imputer,           # Step 3 — fills NaN with median
    credit_assembler,  # Step 4a — wraps cibil_score in vector
    scaler,            # Step 4b — standardizes cibil_score
    *indexers,         # Step 5a — encodes education, self_employed
    label_indexer,     # Step 5b — encodes loan_status → label
    assembler,         # Step 6  — assembles all features into 1 vector
    dt                 # Step 7  — Decision Tree classifier
])""", language="python")


# ── FOOTER ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.8rem; padding:0.5rem 0 1rem;">
  ⚡ Loan Approval Prediction &nbsp;·&nbsp; Apache Spark + PySpark ML &nbsp;·&nbsp;
  Pipeline replicated from <code>loan_approval_spark.py</code> &nbsp;·&nbsp; 2026–27
</div>""", unsafe_allow_html=True)
