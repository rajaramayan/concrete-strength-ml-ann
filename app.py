"""
Concrete Compressive Strength ML & ANN Research — Streamlit App
Compatible with Streamlit ≥ 1.36 (incl. 1.64 on Cloud)
Uses st.navigation / st.Page API — no single-file if/elif page routing.
"""
import sys
import os
import warnings
import traceback

warnings.filterwarnings("ignore")

# ─── Compatibility shim for old scikit-learn gradient boosting pickles ────────
for _mod in ("sklearn._loss.loss", "sklearn._loss"):
    try:
        import importlib
        _m = importlib.import_module(_mod)
        sys.modules.setdefault("_loss", _m)
        break
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

try:
    import keras
except Exception:
    keras = None

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must come before any other st call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Strength ML & ANN",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.hero{background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
      border:1px solid #334155;border-radius:16px;padding:26px 34px;
      margin-bottom:22px;box-shadow:0 10px 25px rgba(0,0,0,.3);}
.hero h1{color:#f8fafc;font-size:2.1rem;font-weight:700;margin:0;}
.hero p {color:#94a3b8;font-size:1rem;margin-top:6px;margin-bottom:0;}
.mcard{background:rgba(30,41,59,.7);border:1px solid #334155;
       border-radius:12px;padding:16px 20px;text-align:center;}
.mval{font-size:1.8rem;font-weight:700;color:#38bdf8;margin-top:4px;}
.mlbl{font-size:.82rem;text-transform:uppercase;letter-spacing:.7px;
      color:#94a3b8;font-weight:600;}
.badge-hi{background:rgba(16,185,129,.15);color:#10b981;padding:4px 12px;
          border-radius:20px;font-size:.84rem;font-weight:600;
          border:1px solid rgba(16,185,129,.3);}
.badge-md{background:rgba(59,130,246,.15);color:#3b82f6;padding:4px 12px;
          border-radius:20px;font-size:.84rem;font-weight:600;
          border:1px solid rgba(59,130,246,.3);}
.badge-lo{background:rgba(245,158,11,.15);color:#f59e0b;padding:4px 12px;
          border-radius:20px;font-size:.84rem;font-weight:600;
          border:1px solid rgba(245,158,11,.3);}
.result-box{background:linear-gradient(135deg,#1e293b,#0f172a);
            border:2px solid #3b82f6;border-radius:14px;padding:22px;
            text-align:center;box-shadow:0 8px 20px rgba(59,130,246,.15);}
.result-val{font-size:2.9rem;font-weight:800;color:#60a5fa;margin:8px 0;}
</style>
"""

def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly helper — works on Streamlit 1.64 (no use_container_width)
# ─────────────────────────────────────────────────────────────────────────────
def _chart(fig, height=None):
    if height:
        fig.update_layout(height=height)
    # Streamlit 1.64: pass width via update_layout; don't use use_container_width
    fig.update_layout(autosize=True)
    st.plotly_chart(fig, use_container_width=True)

def _df(df_in, **kw):
    st.dataframe(df_in, use_container_width=True, **kw)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
]

ALL_MODELS = [
    "XGBoost", "Hybrid XGBoost + ANN", "Random Forest",
    "Gradient Boosting", "Artificial Neural Network", "SVR", "Linear Regression",
]

# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def _load_models():
    """Load all ML models. Returns (models_dict, scaler)."""
    m = {}
    failures = []
    for name, fname in [
        ("XGBoost",           "xgboost.joblib"),
        ("Random Forest",     "random_forest.joblib"),
        ("Gradient Boosting", "gradient_boosting.joblib"),
        ("SVR",               "svr.joblib"),
        ("Linear Regression", "linear_regression.joblib"),
    ]:
        if os.path.exists(fname):
            try:
                m[name] = joblib.load(fname)
            except Exception as e:
                failures.append(f"{name}: {e}")
        else:
            failures.append(f"{name}: file not found ({fname})")

    # ANN
    ann = None
    if keras is not None and os.path.exists("ann_model.keras"):
        try:
            ann_def = keras.Sequential([
                keras.layers.Input(shape=(8,)),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64,  activation="relu"),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32,  activation="relu"),
                keras.layers.Dense(16,  activation="relu"),
                keras.layers.Dense(1,   activation="linear"),
            ])
            ann_def.load_weights("ann_model.keras")
            ann = ann_def
        except Exception as e:
            failures.append(f"ANN keras: {e}")

    if ann is None and os.path.exists("ann_weights.joblib"):
        try:
            ann = joblib.load("ann_weights.joblib")
        except Exception as e:
            failures.append(f"ANN weights: {e}")

    m["Artificial Neural Network"] = ann

    scaler = None
    if os.path.exists("ann_scaler.joblib"):
        try:
            scaler = joblib.load("ann_scaler.joblib")
        except Exception as e:
            failures.append(f"Scaler: {e}")

    return m, scaler, failures


@st.cache_data(show_spinner="Loading datasets…")
def _load_datasets():
    d, failures = {}, []
    for key, fname in [
        ("comparison",         "model_comparison.csv"),
        ("feature_importance", "feature_importance.csv"),
        ("cv",                 "10_fold_cross_validation.csv"),
        ("test_pred",          "test_predictions.csv"),
    ]:
        if os.path.exists(fname):
            try:
                d[key] = pd.read_csv(fname)
            except Exception as e:
                failures.append(f"{fname}: {e}")
        else:
            failures.append(f"{fname}: not found")
    return d, failures


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ann_predict(models, scaler, X_df):
    ann = models.get("Artificial Neural Network")
    X_sc = scaler.transform(X_df) if scaler is not None else X_df.values
    if ann is None:
        return np.zeros(len(X_df))
    if isinstance(ann, list):           # raw numpy weights fallback
        out = X_sc
        for W, b in ann[:-1]:
            out = np.maximum(0, out @ W + b)
        W, b = ann[-1]
        return (out @ W + b).flatten()
    if keras is not None:
        return ann.predict(X_sc, verbose=0).flatten()
    return np.zeros(len(X_df))


def _predict(models, scaler, X_df, model_name):
    if model_name == "Artificial Neural Network":
        return _ann_predict(models, scaler, X_df)
    if model_name == "Hybrid XGBoost + ANN":
        xgb = models.get("XGBoost")
        xp  = xgb.predict(X_df) if xgb else np.zeros(len(X_df))
        ap  = _ann_predict(models, scaler, X_df)
        return (xp + ap) / 2
    mdl = models.get(model_name)
    if mdl is None:
        return np.zeros(len(X_df))
    return mdl.predict(X_df)


def _strength_category(s):
    if s < 20:  return "Low Strength",                    "badge-lo", "Non-structural / footpaths / curbs."
    if s < 40:  return "Standard Structural Concrete",    "badge-md", "Slabs, columns, beams, footings."
    if s < 60:  return "High-Strength Concrete (HSC)",    "badge-hi", "High-rise pillars, pre-stressed girders."
    return      "Ultra-High Performance (UHPC)",           "badge-hi", "Nuclear shielding, extreme marine structures."


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def page_predictor():
    _inject_css()
    try:
        models, scaler, load_fails = _load_models()

        if load_fails:
            with st.expander("⚠️ Model load warnings (click to expand)"):
                for f in load_fails:
                    st.warning(f)

        st.markdown("""
        <div class="hero">
          <h1>🏗️ Concrete Compressive Strength Predictor</h1>
          <p>Simulate concrete mix formulations and predict compressive strength (MPa).</p>
        </div>""", unsafe_allow_html=True)

        # ── Presets ──────────────────────────────────────────────────────────
        PRESETS = {
            "Standard 28D":  (280, 70,  50,  180, 6.0,  980, 770, 28),
            "High-Strength": (450, 100, 0,   150, 12.0, 950, 720, 28),
            "Eco Fly-Ash":   (200, 0,   160, 165, 8.0,  1000,790, 56),
            "Early 7-Day":   (380, 120, 0,   175, 9.0,  920, 750, 7),
        }
        KEYS = ["p_cement","p_slag","p_flyash","p_water","p_sp","p_ca","p_fa","p_age"]
        DEFS = list(PRESETS["Standard 28D"])

        for k, v in zip(KEYS, DEFS):
            if k not in st.session_state:
                st.session_state[k] = v

        def apply_preset(name):
            for k, v in zip(KEYS, PRESETS[name]):
                st.session_state[k] = v

        col_in, col_out = st.columns([1.6, 1.1])

        with col_in:
            st.subheader("⚙️ Mix Parameters")
            st.caption("Quick presets:")
            pc = st.columns(len(PRESETS))
            for i, (nm, _) in enumerate(PRESETS.items()):
                pc[i].button(nm, on_click=apply_preset, args=(nm,), key=f"btn_{i}")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                cement = st.number_input("Cement (kg/m³)",           100.0, 540.0, step=5.0,  key="p_cement")
                slag   = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 360.0, step=5.0,  key="p_slag")
                flyash = st.number_input("Fly Ash (kg/m³)",            0.0, 200.0, step=5.0,  key="p_flyash")
                water  = st.number_input("Water (kg/m³)",            120.0, 250.0, step=2.0,  key="p_water")
            with c2:
                sp     = st.number_input("Superplasticizer (kg/m³)",   0.0,  35.0, step=0.5,  key="p_sp")
                ca     = st.number_input("Coarse Aggregate (kg/m³)",  800.0,1150.0, step=10.0, key="p_ca")
                fa     = st.number_input("Fine Aggregate (kg/m³)",    590.0, 950.0, step=10.0, key="p_fa")
                age    = st.slider("Curing Age (Days)", 1, 365, key="p_age")

            binder = cement + slag + flyash
            wb     = water / binder if binder > 0 else 0
            dens   = binder + water + sp + ca + fa
            m1, m2, m3 = st.columns(3)
            m1.metric("w/b Ratio",    f"{wb:.3f}")
            m2.metric("Total Binder", f"{binder:.1f} kg/m³")
            m3.metric("Mix Density",  f"{dens:.1f} kg/m³")

        with col_out:
            st.subheader("🎯 Prediction Output")
            sel = st.selectbox("Model:", ALL_MODELS, key="p1_model")
            row = pd.DataFrame(
                [[cement, slag, flyash, water, sp, ca, fa, age]],
                columns=FEATURES,
            )
            strength = float(_predict(models, scaler, row, sel)[0])
            cat_title, badge, usage = _strength_category(strength)

            st.markdown(f"""
            <div class="result-box">
              <div style="color:#94a3b8;font-size:.9rem;font-weight:600;">PREDICTED STRENGTH</div>
              <div class="result-val">{strength:.2f}<span style="font-size:1.3rem;"> MPa</span></div>
              <span class="{badge}">{cat_title}</span>
            </div>""", unsafe_allow_html=True)
            st.info(f"💡 **Use case:** {usage}")

            st.markdown("---")
            st.subheader("📊 All-Model Comparison")
            rows = []
            for mn in ALL_MODELS:
                try:
                    v = float(_predict(models, scaler, row, mn)[0])
                except Exception:
                    v = 0.0
                rows.append({"Model": mn, "Strength (MPa)": v})
            pdf = pd.DataFrame(rows)
            fig = px.bar(pdf, x="Strength (MPa)", y="Model", orientation="h",
                         color="Strength (MPa)", color_continuous_scale="Blues",
                         text_auto=".2f")
            fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=10),
                              showlegend=False, yaxis_title=None)
            _chart(fig)

        # ── Batch upload ──────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📁 Batch Prediction — Upload CSV"):
            tmpl = pd.DataFrame(
                [[280,70,50,180,6,980,770,28],[450,100,0,150,12,950,720,28]],
                columns=FEATURES,
            )
            st.download_button("📥 Download template", tmpl.to_csv(index=False),
                               "template.csv", "text/csv", key="dl_tmpl")
            up = st.file_uploader("Upload CSV", type=["csv"], key="batch_up")
            if up:
                bdf  = pd.read_csv(up)
                miss = [c for c in FEATURES if c not in bdf.columns]
                if miss:
                    st.error(f"Missing columns: {miss}")
                else:
                    res = bdf.copy()
                    for mn in ALL_MODELS:
                        try:
                            res[f"{mn} (MPa)"] = np.round(_predict(models, scaler, bdf[FEATURES], mn), 2)
                        except Exception:
                            res[f"{mn} (MPa)"] = np.nan
                    _df(res.head(10))
                    st.download_button("📥 Download results", res.to_csv(index=False),
                                       "batch_results.csv", "text/csv", key="dl_batch")

    except Exception:
        st.error("An error occurred on this page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_models():
    _inject_css()
    try:
        datasets, ds_fails = _load_datasets()
        if ds_fails:
            with st.expander("⚠️ Dataset load warnings"):
                for f in ds_fails:
                    st.warning(f)

        st.markdown("""
        <div class="hero">
          <h1>📊 Model Comparison & Benchmarks</h1>
          <p>MAE, RMSE, R² and 10-Fold Cross-Validation across all models.</p>
        </div>""", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        for col, lbl, val, sub in [
            (k1, "Best R²",     "0.941",    "XGBoost"),
            (k2, "Lowest MAE",  "2.61 MPa", "XGBoost"),
            (k3, "Lowest RMSE", "4.20 MPa", "XGBoost"),
            (k4, "Hybrid R²",   "0.923",    "XGBoost + ANN"),
        ]:
            col.markdown(f"""
            <div class="mcard">
              <div class="mlbl">{lbl}</div>
              <div class="mval">{val}</div>
              <div style="color:#94a3b8;font-size:.78rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🔄 10-Fold CV", "📈 Scatter"])

        with tab1:
            comp = datasets.get("comparison", pd.DataFrame())
            if not comp.empty:
                c1, c2 = st.columns([1.1, 1])
                with c1:
                    _df(comp)
                with c2:
                    fig = px.bar(comp, x="R2", y="Model", orientation="h",
                                 title="R² Score (Higher is Better)",
                                 color="R2", color_continuous_scale="Viridis",
                                 text_auto=".3f")
                    fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
                    _chart(fig, 320)
                fig2 = px.bar(comp, x="Model", y=["MAE", "RMSE"], barmode="group",
                              title="MAE & RMSE (Lower is Better)",
                              color_discrete_sequence=["#38bdf8", "#f43f5e"])
                fig2.update_layout(yaxis_title="Error (MPa)")
                _chart(fig2, 350)
            else:
                st.info("model_comparison.csv not found.")

        with tab2:
            cv = datasets.get("cv", pd.DataFrame())
            if not cv.empty:
                _df(cv)
                fig = go.Figure(go.Bar(
                    x=cv["Model"], y=cv["CV R2 Mean"],
                    error_y=dict(type="data", array=cv["CV R2 Std"], visible=True),
                    marker_color="#6366f1",
                ))
                fig.update_layout(title="10-Fold CV R² ± Std", yaxis_title="Mean R²")
                _chart(fig, 380)
            else:
                st.info("10_fold_cross_validation.csv not found.")

        with tab3:
            tpdf = datasets.get("test_pred", pd.DataFrame())
            if not tpdf.empty:
                fig = px.scatter(
                    tpdf, x="Actual Strength", y="Hybrid Predicted Strength",
                    hover_data=["Cement","Water","Age"],
                    color="Absolute Error", color_continuous_scale="Plasma",
                    title="Actual vs Hybrid Predicted Strength",
                )
                lo = min(tpdf["Actual Strength"].min(), tpdf["Hybrid Predicted Strength"].min())
                hi = max(tpdf["Actual Strength"].max(), tpdf["Hybrid Predicted Strength"].max())
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x",
                                         line=dict(color="#10b981", dash="dash", width=2)))
                _chart(fig, 460)
                fig2 = px.histogram(tpdf, x="Absolute Error", nbins=30,
                                    title="Residual Distribution",
                                    color_discrete_sequence=["#3b82f6"])
                _chart(fig2, 300)
            else:
                st.info("test_predictions.csv not found.")

    except Exception:
        st.error("An error occurred on this page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_features():
    _inject_css()
    try:
        models, scaler, load_fails = _load_models()
        datasets, _               = _load_datasets()

        st.markdown("""
        <div class="hero">
          <h1>🔍 Feature Importance & Sensitivity</h1>
          <p>How each mix parameter drives compressive strength.</p>
        </div>""", unsafe_allow_html=True)

        feat = datasets.get("feature_importance", pd.DataFrame())
        c1, c2 = st.columns([1, 1.3])

        with c1:
            st.subheader("📌 Feature Importance")
            if not feat.empty:
                fsort = feat.sort_values("Importance", ascending=True)
                fig   = px.bar(fsort, x="Importance", y="Feature", orientation="h",
                               color="Importance", color_continuous_scale="Blues",
                               text_auto=".3f")
                fig.update_layout(showlegend=False, yaxis_title=None,
                                  xaxis_title="Importance Score")
                _chart(fig, 430)
                st.info("💡 Age + Cement explain **~66 %** of strength variance.")
            else:
                st.info("feature_importance.csv not found.")

        with c2:
            st.subheader("📈 Strength Growth Simulator")
            sc   = st.slider("Cement (kg/m³)",   150, 500, 300, key="fs_c")
            sw   = st.slider("Water (kg/m³)",     130, 220, 180, key="fs_w")
            ssp  = st.slider("Superplasticizer",  0.0, 20.0, 6.0, key="fs_sp")
            sm   = st.selectbox("Model:", ["XGBoost","Hybrid XGBoost + ANN",
                                           "Random Forest","Artificial Neural Network"],
                                key="fs_m")
            ages = np.arange(1, 181, 2)
            bdf  = pd.DataFrame(
                [[sc, 70, 50, sw, ssp, 950, 750, int(a)] for a in ages],
                columns=FEATURES,
            )
            try:
                y = _predict(models, scaler, bdf, sm)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                y = np.zeros(len(ages))

            fig = px.line(pd.DataFrame({"Age": ages, "Strength": y}),
                          x="Age", y="Strength",
                          title=f"Growth Curve — {sm}", markers=True)
            fig.update_traces(line_color="#38bdf8", line_width=3)
            fig.add_vline(x=28, line_dash="dash", line_color="#10b981",
                          annotation_text="28-Day")
            fig.update_layout(yaxis_title="Strength (MPa)")
            _chart(fig, 370)

        st.markdown("---")
        st.subheader("🌊 Water/Cement Ratio Sensitivity")
        wrange = np.linspace(130, 240, 30)
        wbdf   = pd.DataFrame(
            [[300, 70, 50, w, 6, 950, 750, 28] for w in wrange],
            columns=FEATURES,
        )
        try:
            wy = _predict(models, scaler, wbdf, "XGBoost")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            wy = np.zeros(30)

        fig = px.line(
            pd.DataFrame({"w/c Ratio": wrange / 300, "Strength (MPa)": wy}),
            x="w/c Ratio", y="Strength (MPa)",
            title="Strength vs w/c Ratio — XGBoost", markers=True,
        )
        fig.update_traces(line_color="#f43f5e", line_width=3)
        _chart(fig, 360)

    except Exception:
        st.error("An error occurred on this page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_dataset():
    _inject_css()
    try:
        datasets, ds_fails = _load_datasets()

        st.markdown("""
        <div class="hero">
          <h1>📁 Dataset Explorer & Downloads</h1>
          <p>Filter and download test predictions, CV results, and model metrics.</p>
        </div>""", unsafe_allow_html=True)

        tpdf = datasets.get("test_pred", pd.DataFrame())
        if not tpdf.empty:
            st.subheader("📋 Test Predictions")
            fc1, fc2 = st.columns(2)
            with fc1:
                age_opts   = sorted(tpdf["Age"].unique().tolist())
                age_filter = st.multiselect("Curing Age:", age_opts,
                                            default=[a for a in [7, 28, 90] if a in age_opts],
                                            key="ds_age")
            with fc2:
                smin = float(tpdf["Actual Strength"].min())
                smax = float(tpdf["Actual Strength"].max())
                lo, hi = st.slider("Actual Strength (MPa):", smin, smax,
                                   (smin, smax), key="ds_str")

            mask = (tpdf["Actual Strength"] >= lo) & (tpdf["Actual Strength"] <= hi)
            if age_filter:
                mask &= tpdf["Age"].isin(age_filter)
            _df(tpdf[mask])

        else:
            st.info("test_predictions.csv not found.")

        st.markdown("---")
        st.subheader("📥 Downloads")
        d1, d2, d3 = st.columns(3)

        comp = datasets.get("comparison", pd.DataFrame())
        d1.download_button("Test Predictions",
                           tpdf.to_csv(index=False) if not tpdf.empty else "",
                           "test_predictions.csv", "text/csv", key="dl_tp")
        d2.download_button("Model Comparison",
                           comp.to_csv(index=False) if not comp.empty else "",
                           "model_comparison.csv", "text/csv", key="dl_mc")
        cv = datasets.get("cv", pd.DataFrame())
        d3.download_button("10-Fold CV",
                           cv.to_csv(index=False) if not cv.empty else "",
                           "10_fold_cv.csv", "text/csv", key="dl_cv")

    except Exception:
        st.error("An error occurred on this page:")
        st.code(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────

def page_specs():
    _inject_css()
    try:
        st.markdown("""
        <div class="hero">
          <h1>ℹ️ Research Architecture & Specs</h1>
          <p>ANN architecture, ensemble strategy, training methodology, and dataset details.</p>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("🧠 ANN Architecture")
            st.code("""
Input   :  8 features (concrete mix parameters)
Dense   : 128 units, ReLU
Dense   :  64 units, ReLU
Dropout :  rate = 0.2
Dense   :  32 units, ReLU
Dense   :  16 units, ReLU
Output  :   1 unit,  Linear  → MPa
            """)
            st.markdown("""
**Optimiser:** Adam (lr = 0.001)  
**Loss:** Mean Squared Error  
**Scaler:** StandardScaler on all 8 features  
**Dataset:** 1030 samples, 80/20 train-test split, 10-fold CV  
            """)

        with c2:
            st.subheader("⚡ Hybrid Ensemble")
            st.latex(r"\hat{y}_{hybrid} = 0.5\,\hat{y}_{XGBoost} + 0.5\,\hat{y}_{ANN}")
            data = {
                "Model":          ["XGBoost","Hybrid","GBM","RF","ANN","SVR","Linear Reg."],
                "R²":             [0.941, 0.923, 0.915, 0.910, 0.875, 0.880, 0.580],
                "MAE (MPa)":      [2.61,  2.89,  3.10,  3.24,  4.12,  3.87, 8.91],
                "RMSE (MPa)":     [4.20,  4.71,  4.98,  5.14,  5.93,  5.68,11.30],
            }
            _df(pd.DataFrame(data))

        st.markdown("---")
        st.subheader("📚 Dataset: UCI Concrete Compressive Strength")
        st.markdown("""
| Property | Value |
|---|---|
| Samples | 1030 |
| Features | 8 continuous (cement, slag, fly ash, water, SP, CA, FA, age) |
| Target | Compressive strength (MPa) |
| Age range | 1 – 365 days |
| Strength range | 2.33 – 82.60 MPa |
| Source | I-Cheng Yeh, 1998 (UCI ML Repository) |
        """)
        st.caption("Concrete Compressive Strength ML & ANN Research "
                   "| Streamlit · XGBoost · Scikit-Learn · Keras")

    except Exception:
        st.error("An error occurred on this page:")
        st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════════════════════
# Navigation — Streamlit 1.36+ / 1.64 native API
# ═════════════════════════════════════════════════════════════════════════════
_pages = [
    st.Page(page_predictor, title="Interactive Predictor", icon="🧪", default=True),
    st.Page(page_models,    title="Model Comparison",      icon="📊"),
    st.Page(page_features,  title="Feature Analysis",      icon="🔍"),
    st.Page(page_dataset,   title="Dataset Explorer",      icon="📁"),
    st.Page(page_specs,     title="Research Specs",        icon="ℹ️"),
]

pg = st.navigation(_pages, position="sidebar")
pg.run()
