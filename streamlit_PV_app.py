import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Model Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f97316 0%, #fbbf24 60%, #facc15 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #8b949e;
    font-size: 1rem;
    margin-bottom: 1.8rem;
    font-weight: 300;
}
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f97316;
    border-left: 3px solid #f97316;
    padding-left: 0.7rem;
    margin: 1.4rem 0 0.8rem;
}
.step-pill {
    display: inline-block;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    color: #f97316;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.mcard {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    transition: border-color .2s;
}
.mcard:hover { border-color: #f97316; }
.mcard.best  { border-left: 4px solid #f97316; background: #1a1f27; }
.mcard-name  {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0.6rem;
}
.best-badge {
    background: #f97316;
    color: #0d1117;
    font-size: 0.65rem;
    font-weight: 800;
    padding: 2px 9px;
    border-radius: 99px;
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: .06em;
}
.metrics-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }
.metric-cell { text-align: center; }
.metric-val  { font-size: 1.25rem; font-weight: 700; color: #f0f6fc; }
.metric-lbl  { font-size: 0.68rem; color: #6e7681; text-transform: uppercase; letter-spacing: .05em; }
.winner {
    background: linear-gradient(90deg,#f97316,#fbbf24);
    border-radius: 12px;
    padding: 0.9rem 1.5rem;
    color: #0d1117;
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 1.2rem;
}
.pred-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.pred-val { font-size: 2rem; font-weight: 800; }
.pred-model { font-size: 0.78rem; color: #8b949e; margin-top: 3px; }
button[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #8b949e !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f97316 !important;
    border-bottom-color: #f97316 !important;
}
[data-testid="stMetricValue"] { color: #f0f6fc; font-family: 'Syne', sans-serif; }
[data-testid="stMetricLabel"] { color: #6e7681; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
MODEL_META = {
    "Linear Regression": {
        "icon": "📈", "color": "#e74c3c", "needs_scale": True,
        "desc": "Fits a linear hyperplane via Ordinary Least Squares. Fast baseline model.",
    },
    "Gradient Descent (SGD)": {
        "icon": "⚡", "color": "#3b82f6", "needs_scale": True,
        "desc": "Linear regression solved iteratively using Stochastic Gradient Descent.",
    },
    "Lasso Regression": {
        "icon": "🔶", "color": "#a855f7", "needs_scale": True,
        "desc": "L1 regularisation — shrinks unimportant coefficients to exactly zero.",
    },
    "Ridge Regression": {
        "icon": "🔷", "color": "#06b6d4", "needs_scale": True,
        "desc": "L2 regularisation — shrinks all coefficients smoothly to reduce overfitting.",
    },
    "Decision Tree": {
        "icon": "🌳", "color": "#10b981", "needs_scale": False,
        "desc": "Recursive binary splits on features. Interpretable but can overfit.",
    },
    "Random Forest": {
        "icon": "🌲", "color": "#f97316", "needs_scale": False,
        "desc": "Ensemble of decision trees with bagging + random feature subsets.",
    },
}

PARAM_GRIDS = {
    "Linear Regression": {},
    "Gradient Descent (SGD)": {
        "eta0": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "max_iter": [500, 1000, 2000],
    },
    "Lasso Regression":  {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "Ridge Regression":  {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "Decision Tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    },
}

SENS_PARAMS = {
    "Lasso Regression":       ("alpha",        [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]),
    "Ridge Regression":       ("alpha",        [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]),
    "Gradient Descent (SGD)": ("eta0",         [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]),
    "Decision Tree":          ("max_depth",    [2, 3, 5, 8, 12, 20]),
    "Random Forest":          ("n_estimators", [10, 30, 50, 100, 200, 300]),
}

PLOTLY_DARK = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    font=dict(family="DM Sans", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
)

MPL_RC = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",      "ytick.color": "#8b949e",
    "grid.color": "#21262d",       "axes.titlecolor": "#e6edf3",
    "axes.labelcolor": "#8b949e",
}

# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_file(file_bytes, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == "csv":
        return pd.read_csv(buf)
    if ext in ("xls", "xlsx"):
        return pd.read_excel(buf)
    return None


def preprocess(df, features, target):
    sub = df[features + [target]].copy()
    for c in sub.select_dtypes(include="object").columns:
        sub[c] = LabelEncoder().fit_transform(sub[c].astype(str))
    return sub.dropna()


def build_model(name, params):
    md = params.get("max_depth", 10)
    md = None if md == 0 else md
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Gradient Descent (SGD)":
        return SGDRegressor(eta0=params.get("eta0", 0.01),
                            learning_rate=params.get("lr_schedule", "invscaling"),
                            max_iter=params.get("max_iter", 1000), random_state=42)
    if name == "Lasso Regression":
        return Lasso(alpha=params.get("alpha", 1.0), max_iter=10000)
    if name == "Ridge Regression":
        return Ridge(alpha=params.get("alpha", 1.0))
    if name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=md,
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1), random_state=42)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=params.get("n_estimators", 100),
            max_depth=md, min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42, n_jobs=-1)


def train_one(name, params, Xtr, ytr, Xte, yte, scaler):
    sc  = MODEL_META[name]["needs_scale"]
    Xtr_ = scaler.transform(Xtr) if sc else Xtr.values
    Xte_ = scaler.transform(Xte) if sc else Xte.values
    m = build_model(name, params)
    m.fit(Xtr_, ytr)
    ptr = m.predict(Xtr_)
    pte = m.predict(Xte_)
    cv  = cross_val_score(m, Xtr_, ytr, cv=5, scoring="r2").mean()
    return {
        "model": m, "needs_scale": sc,
        "pred_tr": ptr, "pred_te": pte,
        "r2_tr": r2_score(ytr, ptr), "r2_te": r2_score(yte, pte),
        "rmse": np.sqrt(mean_squared_error(yte, pte)),
        "mae": mean_absolute_error(yte, pte),
        "cv_r2": cv,
        "color": MODEL_META[name]["color"],
        "icon":  MODEL_META[name]["icon"],
    }


def run_gs(name, Xtr_s, ytr):
    base = build_model(name, {})
    grid = PARAM_GRIDS[name]
    if not grid:
        base.fit(Xtr_s, ytr)
        cv = cross_val_score(base, Xtr_s, ytr, cv=5, scoring="r2")
        return base, {}, cv.mean(), cv
    gs = GridSearchCV(base, grid, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(Xtr_s, ytr)
    cv = cross_val_score(gs.best_estimator_, Xtr_s, ytr, cv=5, scoring="r2")
    return gs.best_estimator_, gs.best_params_, gs.best_score_, cv


# ─────────────────────────────────────────────────────────────
# MATPLOTLIB CHARTS
# ─────────────────────────────────────────────────────────────
def mpl_avp(results, y_test):
    srt = sorted(results.items(), key=lambda x: x[1]["r2_te"], reverse=True)
    n = len(srt); cols = 3; rows = (n + 2) // 3
    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4.5))
        axes = np.array(axes).flatten()
        for i, (nm, r) in enumerate(srt):
            ax = axes[i]
            ax.scatter(y_test, r["pred_te"], c=r["color"], alpha=0.55, s=35, linewidths=0)
            lo = min(float(np.min(y_test)), float(r["pred_te"].min()))
            hi = max(float(np.max(y_test)), float(r["pred_te"].max()))
            ax.plot([lo, hi], [lo, hi], "w--", lw=1.4, alpha=0.6)
            ax.set_title(f"{r['icon']} {nm}\nR²={r['r2_te']:.4f}  RMSE={r['rmse']:.1f}",
                         fontsize=9.5, fontweight="bold", pad=6)
            ax.set_xlabel("Actual", fontsize=8); ax.set_ylabel("Predicted", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(True, alpha=0.4, lw=0.5)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        plt.suptitle("Actual vs Predicted — All Models", fontsize=13, fontweight="bold",
                     color="#e6edf3", y=1.01)
        plt.tight_layout()
    return fig


def mpl_resid(results, y_test):
    srt = sorted(results.items(), key=lambda x: x[1]["r2_te"], reverse=True)
    n = len(srt); cols = 3; rows = (n + 2) // 3
    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = np.array(axes).flatten()
        for i, (nm, r) in enumerate(srt):
            ax = axes[i]
            resid = np.array(y_test) - r["pred_te"]
            ax.hist(resid, bins=25, color=r["color"], alpha=0.75, edgecolor="#0d1117", lw=0.5)
            ax.axvline(0, color="white", lw=1.4, ls="--", alpha=0.7)
            ax.set_title(f"{r['icon']} {nm}\nMean={resid.mean():.2f}  Std={resid.std():.2f}",
                         fontsize=9.5, fontweight="bold", pad=6)
            ax.set_xlabel("Residual", fontsize=8); ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(True, alpha=0.4, lw=0.5)
        for j in range(i+1, len(axes)): axes[j].set_visible(False)
        plt.suptitle("Residual Distributions — All Models", fontsize=13, fontweight="bold",
                     color="#e6edf3", y=1.01)
        plt.tight_layout()
    return fig


def mpl_feat_imp(results, feat_cols):
    tree_r = {n: v for n, v in results.items()
              if hasattr(v["model"], "feature_importances_")}
    if not tree_r: return None
    n = len(tree_r)
    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(1, n, figsize=(8*n, 5))
        if n == 1: axes = [axes]
        for ax, (nm, r) in zip(axes, tree_r.items()):
            imp  = r["model"].feature_importances_
            idx  = np.argsort(imp)[::-1][:12]
            lbls = [feat_cols[i] for i in idx]
            ax.barh(lbls[::-1], imp[idx][::-1], color=r["color"], edgecolor="#0d1117")
            ax.set_title(f"{r['icon']} {nm}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Importance", fontsize=8)
            ax.tick_params(labelsize=7.5); ax.grid(True, axis="x", alpha=0.4, lw=0.5)
        plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────
def plotly_metrics(results):
    srt    = sorted(results.items(), key=lambda x: x[1]["r2_te"], reverse=True)
    names  = [n for n, _ in srt]
    colors = [v["color"] for _, v in srt]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("R² Score ↑", "RMSE ↓", "MAE ↓"))
    for ci, key in enumerate(["r2_te", "rmse", "mae"], 1):
        vals = [results[n][key] for n in names]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors,
                             text=[f"{v:.3f}" for v in vals], textposition="outside",
                             textfont=dict(size=10), showlegend=False), row=1, col=ci)
    fig.update_layout(height=380, **PLOTLY_DARK)
    fig.update_annotations(font_color="#f97316", font_size=12)
    fig.update_xaxes(tickangle=30, tickfont_size=9, gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    return fig


def plotly_sens(name, pname, pvals, Xtr_s, ytr, Xte_s, yte):
    tr_r, te_r = [], []
    for v in pvals:
        m = build_model(name, {pname: v})
        m.fit(Xtr_s, ytr)
        tr_r.append(r2_score(ytr, m.predict(Xtr_s)))
        te_r.append(r2_score(yte, m.predict(Xte_s)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[str(v) for v in pvals], y=tr_r, mode="lines+markers",
                             name="Train R²", line=dict(color="#fbbf24", width=2),
                             marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=[str(v) for v in pvals], y=te_r, mode="lines+markers",
                             name="Test R²", line=dict(color="#f97316", width=2),
                             marker=dict(size=7)))
    fig.update_layout(title=f"Sensitivity — {pname}", xaxis_title=pname,
                      yaxis_title="R² Score", height=380, **PLOTLY_DARK)
    return fig


def plotly_cv_bar(cv_scores, color, name):
    fig = go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(len(cv_scores))], y=cv_scores,
        marker_color=color, text=[f"{v:.4f}" for v in cv_scores],
        textposition="outside"))
    fig.update_layout(title=f"5-Fold CV R² — {name}", yaxis_title="R²",
                      height=350, **PLOTLY_DARK)
    return fig


def plotly_scatter_tuned(y_test, pred, color, name):
    lo = min(float(np.min(y_test)), float(pred.min()))
    hi = max(float(np.max(y_test)), float(pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(y_test), y=list(pred), mode="markers",
                             marker=dict(color=color, size=7, opacity=0.6), name="Predictions"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="white", dash="dash", width=1.5), name="Perfect"))
    fig.update_layout(title=f"Actual vs Predicted — {name}", xaxis_title="Actual",
                      yaxis_title="Predicted", height=420, **PLOTLY_DARK)
    return fig


def plotly_pred_bar(preds, results):
    names  = list(preds.keys())
    vals   = list(preds.values())
    colors = [results[n]["color"] for n in names]
    avg    = np.mean(vals)
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                           text=[f"{v:.2f}" for v in vals], textposition="outside",
                           textfont=dict(size=11)))
    fig.add_hline(y=avg, line_dash="dot", line_color="white",
                  annotation_text=f"avg = {avg:.2f}",
                  annotation_font=dict(color="white", size=10))
    fig.update_layout(title="All-Model Prediction Comparison", xaxis_title="Model",
                      yaxis_title="Predicted Value", showlegend=False,
                      height=380, **PLOTLY_DARK)
    fig.update_xaxes(tickangle=25)
    return fig


# ─────────────────────────────────────────────────────────────
# ════════════════════  MAIN APP  ════════════════════════════
# ─────────────────────────────────────────────────────────────
def main():

    # HERO
    st.markdown('<div class="hero-title">🧪 ML Model Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Upload any dataset · Choose your target & features · '
        'Train  models and hypertune with Grid Search</div>',
        unsafe_allow_html=True)

    # ════════════════  SIDEBAR  ══════════════════════════════
    sb = st.sidebar
    sb.markdown("## ⚙️ Configuration")
    sb.markdown("---")

    # STEP 1 — Upload
    sb.markdown('<span class="step-pill">STEP 1 — Upload Dataset</span>',
                unsafe_allow_html=True)
    uploaded = sb.file_uploader("CSV or Excel file", type=["csv","xlsx","xls"])

    if not uploaded:
        # welcome screen
        st.markdown('<div class="section-head">👋 Getting Started</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, step, icon, txt in zip([c1,c2,c3],
            ["Step 1","Step 2","Step 3"],
            ["📁","🎯","🚀"],
            ["Upload any CSV/Excel file",
             "Choose target & feature columns with checkboxes",
             "Train all 6 models & hypertune"]):
            col.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                        padding:1.4rem;text-align:center">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;
                            color:#f97316;margin:.4rem 0 .3rem">{step}</div>
                <div style="color:#8b949e;font-size:.9rem">{txt}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div class="section-head">🤖 6 ML Models Included</div>',
                    unsafe_allow_html=True)
        for nm, m in MODEL_META.items():
            st.markdown(f"**{m['icon']} {nm}** — "
                        f"<span style='color:#8b949e'>{m['desc']}</span>",
                        unsafe_allow_html=True)
        st.stop()

    # Load
    raw = load_file(uploaded.getvalue(), uploaded.name)
    if raw is None:
        sb.error("Could not read file."); st.stop()

    # clean column names (strip spaces)
    raw.columns = raw.columns.str.strip()
    sb.success(f"✅ {len(raw):,} rows · {len(raw.columns)} columns")

    # STEP 2 — Target
    sb.markdown("---")
    sb.markdown('<span class="step-pill">STEP 2 — Target Column</span>',
                unsafe_allow_html=True)
    all_cols = raw.columns.tolist()
    num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    kws = ["power","output","target","price","sales","energy","generation",
           "main","consumption","yield","revenue","demand"]
    default_tgt = next((c for c in num_cols if any(k in c.lower() for k in kws)), num_cols[-1])
    target_col = sb.selectbox("Column to predict",
                               all_cols, index=all_cols.index(default_tgt))

    # STEP 3 — Features
    sb.markdown("---")
    sb.markdown('<span class="step-pill">STEP 3 — Feature Columns</span>',
                unsafe_allow_html=True)
    avail = [c for c in num_cols if c != target_col]

    bc1, bc2 = sb.columns(2)
    if bc1.button("✅ All",  use_container_width=True):
        st.session_state["sel_feats"] = avail[:]
    if bc2.button("🗑 Clear", use_container_width=True):
        st.session_state["sel_feats"] = []
    if "sel_feats" not in st.session_state:
        st.session_state["sel_feats"] = avail[:]

    sel_feats = [f for f in avail
                 if sb.checkbox(f, value=(f in st.session_state["sel_feats"]), key=f"cb_{f}")]
    if not sel_feats:
        sb.warning("Select at least 1 feature."); st.stop()
    sb.info(f"✅ {len(sel_feats)} features selected")

    # STEP 4 — Models
    sb.markdown("---")
    sb.markdown('<span class="step-pill">STEP 4 — Models</span>', unsafe_allow_html=True)
    sel_models = [nm for nm in MODEL_META
                  if sb.checkbox(nm, value=True, key=f"m_{nm}")]
    if not sel_models:
        sb.warning("Select at least 1 model."); st.stop()

    # STEP 5 — Split
    sb.markdown("---")
    sb.markdown('<span class="step-pill">STEP 5 — Split & Seed</span>',
                unsafe_allow_html=True)
    test_pct  = sb.slider("Test size (%)", 10, 40, 20)
    rand_seed = sb.number_input("Random seed", 0, 9999, 42)

    # Manual HPs
    sb.markdown("---")
    sb.markdown('<span class="step-pill">OPTIONAL — Manual Hyperparameters</span>',
                unsafe_allow_html=True)
    sb.caption("Adjust before training (or use Grid Search in Hypertune tab).")
    hp = {}
    if "Gradient Descent (SGD)" in sel_models:
        with sb.expander("⚡ SGD"):
            hp["Gradient Descent (SGD)"] = {
                "eta0":      sb.slider("Learning rate", 0.0001, 1.0, 0.01, format="%.4f", key="sgd_e"),
                "max_iter":  sb.slider("Max iterations", 100, 5000, 1000, key="sgd_i"),
                "lr_schedule": sb.selectbox("LR schedule",["invscaling","constant","adaptive"], key="sgd_s"),
            }
    if "Lasso Regression" in sel_models:
        with sb.expander("🔶 Lasso"):
            hp["Lasso Regression"] = {"alpha": sb.select_slider("Alpha",
                [0.0001,0.001,0.01,0.1,1.0,10.0,100.0], value=1.0, key="la_a")}
    if "Ridge Regression" in sel_models:
        with sb.expander("🔷 Ridge"):
            hp["Ridge Regression"] = {"alpha": sb.select_slider("Alpha",
                [0.0001,0.001,0.01,0.1,1.0,10.0,100.0], value=1.0, key="ri_a")}
    if "Decision Tree" in sel_models:
        with sb.expander("🌳 Decision Tree"):
            hp["Decision Tree"] = {
                "max_depth":         sb.slider("Max depth (0=none)", 0, 30, 10, key="dt_d"),
                "min_samples_split": sb.slider("Min samples split",  2, 20,  2, key="dt_s"),
                "min_samples_leaf":  sb.slider("Min samples leaf",   1, 20,  1, key="dt_l"),
            }
    if "Random Forest" in sel_models:
        with sb.expander("🌲 Random Forest"):
            hp["Random Forest"] = {
                "n_estimators":      sb.slider("Trees",              10, 500, 100, key="rf_n"),
                "max_depth":         sb.slider("Max depth (0=none)", 0,  30,  10,  key="rf_d"),
                "min_samples_split": sb.slider("Min samples split",  2,  20,   2,  key="rf_s"),
                "min_samples_leaf":  sb.slider("Min samples leaf",   1,  20,   1,  key="rf_l"),
            }

    sb.markdown("---")
    train_btn = sb.button("🚀 Train All Models", type="primary", use_container_width=True)

    # PREPARE DATA
    df_clean = preprocess(raw, sel_feats, target_col)
    X = df_clean[sel_feats];  y = df_clean[target_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct/100,
                                               random_state=rand_seed)
    scaler = StandardScaler().fit(X_tr)

    # TOP METRICS
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Rows", f"{len(df_clean):,}")
    c2.metric("Train",      f"{len(X_tr):,}")
    c3.metric("Test",       f"{len(X_te):,}")
    c4.metric("Features",   len(sel_feats))
    c5.metric("Models",     len(sel_models))

    # TRAIN
    if train_btn:
        results = {}
        prog = st.progress(0); status = st.empty()
        for i, nm in enumerate(sel_models):
            status.text(f"⏳ Training {nm} …")
            results[nm] = train_one(nm, hp.get(nm, {}), X_tr, y_tr, X_te, y_te, scaler)
            prog.progress((i+1)/len(sel_models))
        prog.empty(); status.empty()
        st.session_state.update({
            "results": results,
            "y_te": y_te.values, "y_tr": y_tr.values,
            "X_tr": X_tr, "X_te": X_te,
            "scaler": scaler,
            "sel_feats": sel_feats,
            "target_col": target_col,
            "df_clean": df_clean,
        })
        st.success("✅ All models trained! Browse the tabs below.")
        st.balloons()

    # ════════════════  TABS  ═════════════════════════════════
    tab_data, tab_res, tab_pred, tab_tune = st.tabs([
        "📊 Data Explorer",
        "🏆 Model Results",
        "🔮 Predict",
        "🎛️ Hypertune",
    ])

    # ── TAB 1 : DATA EXPLORER ────────────────────────────────
    with tab_data:
        st.markdown('<div class="section-head">📋 Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(raw[sel_feats + [target_col]].head(30), use_container_width=True)

        st.markdown('<div class="section-head">📊 Statistics</div>', unsafe_allow_html=True)
        st.dataframe(raw[sel_feats + [target_col]].describe().round(3),
                     use_container_width=True)

        st.markdown('<div class="section-head">🔥 Correlation Heatmap</div>',
                    unsafe_allow_html=True)
        show = sel_feats[:14] + [target_col]
        corr = raw[show].corr()
        fig_h = px.imshow(corr, text_auto=".2f",
                          color_continuous_scale="RdYlGn", zmin=-1, zmax=1, aspect="auto")
        fig_h.update_layout(height=500, **PLOTLY_DARK,
                            coloraxis_colorbar=dict(tickfont=dict(color="#c9d1d9")))
        fig_h.update_traces(textfont_size=9)
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown('<div class="section-head">📈 Target Distribution</div>',
                    unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            fd = px.histogram(raw, x=target_col, nbins=40,
                              color_discrete_sequence=["#f97316"])
            fd.update_layout(height=320, **PLOTLY_DARK,
                             xaxis_title=target_col, yaxis_title="Count")
            st.plotly_chart(fd, use_container_width=True)
        with cb:
            xf = st.selectbox("Feature for scatter", sel_feats, key="sc_x")
            fs = px.scatter(raw, x=xf, y=target_col,
                            color_discrete_sequence=["#3b82f6"], opacity=0.55)
            fs.update_layout(height=320, **PLOTLY_DARK)
            st.plotly_chart(fs, use_container_width=True)

    # ── TAB 2 : MODEL RESULTS ────────────────────────────────
    with tab_res:
        if "results" not in st.session_state:
            st.info("👈 Train models first."); st.stop()

        res  = st.session_state["results"]
        y_te = st.session_state["y_te"]
        srt  = sorted(res.items(), key=lambda x: x[1]["r2_te"], reverse=True)
        best = srt[0][0]
        b    = res[best]

        # winner
        st.markdown(
            f'<div class="winner">🏆 Best: {b["icon"]} {best} &nbsp;|&nbsp; '
            f'R² = {b["r2_te"]:.4f} &nbsp;|&nbsp; RMSE = {b["rmse"]:.2f} &nbsp;|&nbsp; '
            f'MAE = {b["mae"]:.2f}</div>',
            unsafe_allow_html=True)

        # cards
        st.markdown('<div class="section-head">📋 All Model Cards</div>',
                    unsafe_allow_html=True)
        for nm, r in srt:
            badge = '<span class="best-badge">BEST</span>' if nm == best else ""
            cls   = "mcard best" if nm == best else "mcard"
            st.markdown(f"""
            <div class="{cls}">
              <div class="mcard-name">{r['icon']} {nm}{badge}</div>
              <div class="metrics-row">
                <div class="metric-cell">
                  <div class="metric-val">{r['r2_te']:.4f}</div>
                  <div class="metric-lbl">R² Test</div>
                </div>
                <div class="metric-cell">
                  <div class="metric-val">{r['r2_tr']:.4f}</div>
                  <div class="metric-lbl">R² Train</div>
                </div>
                <div class="metric-cell">
                  <div class="metric-val">{r['rmse']:.2f}</div>
                  <div class="metric-lbl">RMSE</div>
                </div>
                <div class="metric-cell">
                  <div class="metric-val">{r['mae']:.2f}</div>
                  <div class="metric-lbl">MAE</div>
                </div>
              </div>
              <div style="margin-top:.5rem;color:#6e7681;font-size:.78rem">
                CV R² (5-fold): {r['cv_r2']:.4f} &nbsp;·&nbsp;
                Overfit gap: {r['r2_tr']-r['r2_te']:.4f}
              </div>
            </div>""", unsafe_allow_html=True)

        # plotly bars
        st.markdown('<div class="section-head">📊 Metric Comparison</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plotly_metrics(res), use_container_width=True)

        # matplotlib grids
        st.markdown('<div class="section-head">📈 Actual vs Predicted — All Models</div>',
                    unsafe_allow_html=True)
        fig_ap = mpl_avp(res, y_te)
        st.pyplot(fig_ap, use_container_width=True); plt.close()

        st.markdown('<div class="section-head">📉 Residual Distributions</div>',
                    unsafe_allow_html=True)
        fig_rd = mpl_resid(res, y_te)
        st.pyplot(fig_rd, use_container_width=True); plt.close()

        st.markdown('<div class="section-head">🔍 Feature Importance (Tree Models)</div>',
                    unsafe_allow_html=True)
        fi = mpl_feat_imp(res, st.session_state["sel_feats"])
        if fi:
            st.pyplot(fi, use_container_width=True); plt.close()
        else:
            st.info("No tree-based models selected.")

        # summary table
        st.markdown('<div class="section-head">📋 Summary Table</div>',
                    unsafe_allow_html=True)
        tbl = pd.DataFrame([{
            "Model":         f"{v['icon']} {n}",
            "R² (Test)":     round(v["r2_te"], 4),
            "R² (Train)":    round(v["r2_tr"], 4),
            "RMSE":          round(v["rmse"],  2),
            "MAE":           round(v["mae"],   2),
            "CV R² (5-fold)":round(v["cv_r2"],4),
            "Overfit Gap":   round(v["r2_tr"]-v["r2_te"], 4),
        } for n, v in srt])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        # descriptions
        st.markdown('<div class="section-head">📚 Model Descriptions</div>',
                    unsafe_allow_html=True)
        for nm in res:
            m = MODEL_META[nm]
            st.markdown(f"**{m['icon']} {nm}** — "
                        f"<span style='color:#8b949e'>{m['desc']}</span>",
                        unsafe_allow_html=True)

    # ── TAB 3 : PREDICT ──────────────────────────────────────
    with tab_pred:
        if "results" not in st.session_state:
            st.info("👈 Train models first."); st.stop()

        res     = st.session_state["results"]
        feats   = st.session_state["sel_feats"]
        df_c    = st.session_state["df_clean"]
        scl     = st.session_state["scaler"]
        tgt_lbl = st.session_state["target_col"]
        srt     = sorted(res.items(), key=lambda x: x[1]["r2_te"], reverse=True)

        st.markdown('<div class="section-head">🎛️ Adjust Feature Values</div>',
                    unsafe_allow_html=True)
        st.caption("Move sliders — all 6 models predict instantly.")

        inp = {}
        cols3 = st.columns(3)
        for i, f in enumerate(feats):
            lo = float(df_c[f].min()); hi = float(df_c[f].max()); mn = float(df_c[f].mean())
            with cols3[i % 3]:
                inp[f] = st.slider(f, lo, hi, mn, key=f"sl_{f}") \
                         if lo != hi else st.number_input(f, value=mn, key=f"ni_{f}")

        X_in = pd.DataFrame([[inp[f] for f in feats]], columns=feats)
        preds = {}
        for nm, r in res.items():
            x = scl.transform(X_in) if r["needs_scale"] else X_in.values
            preds[nm] = max(0.0, float(r["model"].predict(x)[0]))

        st.markdown('<div class="section-head">🔮 Predictions from All Models</div>',
                    unsafe_allow_html=True)
        pcols = st.columns(len(preds))
        for col_ui, (nm, val) in zip(pcols, srt):
            r = res[nm]
            col_ui.markdown(f"""
            <div class="pred-card" style="border-color:{r['color']}40">
              <div style="font-size:1.6rem">{r['icon']}</div>
              <div class="pred-val" style="color:{r['color']}">{val:.2f}</div>
              <div class="pred-model">{nm}</div>
              <div style="font-size:.7rem;color:#6e7681;margin-top:4px">R² {r['r2_te']:.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-head">📊 Comparison Chart</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(plotly_pred_bar(preds, res), use_container_width=True)

        avg = np.mean(list(preds.values()))
        best_nm = srt[0][0]
        s1,s2,s3,s4 = st.columns(4)
        s1.metric(f"🏆 {best_nm}", f"{preds[best_nm]:.2f}")
        s2.metric("📊 Average",    f"{avg:.2f}")
        s3.metric("⬆️ Max",        f"{max(preds.values()):.2f}")
        s4.metric("⬇️ Min",        f"{min(preds.values()):.2f}")

    # ── TAB 4 : HYPERTUNE ────────────────────────────────────
    with tab_tune:
        if "results" not in st.session_state:
            st.info("👈 Train models first."); st.stop()

        res   = st.session_state["results"]
        y_te  = st.session_state["y_te"]
        y_tr  = st.session_state["y_tr"]
        X_tr_ = st.session_state["X_tr"]
        X_te_ = st.session_state["X_te"]
        scl   = st.session_state["scaler"]

        st.markdown('<div class="section-head">🎛️ Select Model to Tune</div>',
                    unsafe_allow_html=True)
        tune_nm   = st.selectbox("Model", list(res.keys()), key="tune_sel")
        meta      = MODEL_META[tune_nm]
        needs_sc  = meta["needs_scale"]
        Xtr_s = scl.transform(X_tr_) if needs_sc else X_tr_.values
        Xte_s = scl.transform(X_te_) if needs_sc else X_te_.values

        # grid display
        grid = PARAM_GRIDS.get(tune_nm, {})
        if grid:
            st.markdown("**Grid Search parameter space:**")
            for k, vals in grid.items():
                st.markdown(f"- `{k}` → `{vals}`")
        else:
            st.info("Linear Regression has no tunable hyperparameters.")

        # sensitivity
        if tune_nm in SENS_PARAMS:
            st.markdown('<div class="section-head">📈 Parameter Sensitivity</div>',
                        unsafe_allow_html=True)
            pname, pvals = SENS_PARAMS[tune_nm]
            st.caption(f"How does **{pname}** affect train vs test R²?")
            with st.spinner("Computing sensitivity …"):
                fig_sens = plotly_sens(tune_nm, pname, pvals,
                                       Xtr_s, y_tr, Xte_s, y_te)
            st.plotly_chart(fig_sens, use_container_width=True)

        # grid search
        st.markdown("---")
        gs_btn = st.button("🔍 Run Grid Search — Find Best Params",
                           type="primary", key="gs_run")

        if gs_btn:
            if not grid:
                st.info("No parameters to search for this model.")
            else:
                with st.spinner(f"Grid search running for {tune_nm} … may take 1–2 min"):
                    best_est, best_p, best_cv, cv_all = run_gs(tune_nm, Xtr_s, y_tr)

                st.success("✅ Done!")

                if best_p:
                    st.markdown("**🏆 Best Parameters Found:**")
                    st.dataframe(
                        pd.DataFrame({"Parameter": list(best_p.keys()),
                                      "Best Value": list(best_p.values())}),
                        use_container_width=True, hide_index=True)

                st.markdown('<div class="section-head">📊 CV Fold Scores</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plotly_cv_bar(cv_all, meta["color"], tune_nm),
                                use_container_width=True)

                # test performance
                y_pred_t = best_est.predict(Xte_s)
                r2_t  = r2_score(y_te, y_pred_t)
                rmse_t = np.sqrt(mean_squared_error(y_te, y_pred_t))
                mae_t  = mean_absolute_error(y_te, y_pred_t)

                st.markdown('<div class="section-head">📋 Tuned Model Performance</div>',
                            unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("R² (Test)", f"{r2_t:.4f}")
                m2.metric("RMSE",      f"{rmse_t:.2f}")
                m3.metric("MAE",       f"{mae_t:.2f}")
                m4.metric("CV R²",     f"{best_cv:.4f}")

                # delta vs manual
                prev = res[tune_nm]
                st.markdown("**vs. current manual settings:**")
                d1,d2,d3 = st.columns(3)
                dr = r2_t - prev["r2_te"]; drm = rmse_t - prev["rmse"]
                d1.metric("ΔR²",   f"{dr:+.4f}",
                          delta_color="normal" if dr >= 0 else "inverse")
                d2.metric("ΔRMSE", f"{drm:+.2f}",
                          delta_color="inverse" if drm < 0 else "normal")
                d3.metric("ΔMAE",  f"{mae_t-prev['mae']:+.2f}",
                          delta_color="inverse" if mae_t < prev["mae"] else "normal")

                st.markdown('<div class="section-head">📈 Actual vs Predicted (Tuned)</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(plotly_scatter_tuned(y_te, y_pred_t, meta["color"],
                                                     f"Tuned {tune_nm}"),
                                use_container_width=True)

    st.markdown(
        "<div style='text-align:center;color:#30363d;font-size:.8rem;margin-top:2rem'>"
        "🧪 ML Model Lab · 6 Models · Upload Any Dataset · Plotly + Matplotlib</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
