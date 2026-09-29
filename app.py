import sys
import os
import warnings

# Alias legacy unpickling module for Gradient Boosting compatibility
try:
    import sklearn._loss.loss
    sys.modules['_loss'] = sklearn._loss.loss
except Exception:
    try:
        import sklearn._loss
        sys.modules['_loss'] = sklearn._loss
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib

try:
    import keras
except ImportError:
    keras = None

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Strength ML & ANN Research",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-banner {
    background: linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
    border:1px solid #334155; border-radius:16px;
    padding:28px 36px; margin-bottom:25px;
    box-shadow:0 10px 25px rgba(0,0,0,.3);
}
.hero-title  { color:#f8fafc; font-size:2.2rem; font-weight:700; margin:0; }
.hero-subtitle { color:#94a3b8; font-size:1.05rem; margin-top:8px; margin-bottom:0; }
.metric-card {
    background:rgba(30,41,59,.7); border:1px solid #334155;
    border-radius:12px; padding:18px 22px; text-align:center;
}
.metric-value { font-size:1.8rem; font-weight:700; color:#38bdf8; margin-top:4px; }
.metric-label { font-size:.85rem; text-transform:uppercase; letter-spacing:.8px; color:#94a3b8; font-weight:600; }
.badge-high    { background:rgba(16,185,129,.15); color:#10b981; padding:4px 12px; border-radius:20px; font-size:.85rem; font-weight:600; border:1px solid rgba(16,185,129,.3); }
.badge-std     { background:rgba(59,130,246,.15);  color:#3b82f6; padding:4px 12px; border-radius:20px; font-size:.85rem; font-weight:600; border:1px solid rgba(59,130,246,.3); }
.badge-warning { background:rgba(245,158,11,.15); color:#f59e0b; padding:4px 12px; border-radius:20px; font-size:.85rem; font-weight:600; border:1px solid rgba(245,158,11,.3); }
.result-box {
    background:linear-gradient(135deg,#1e293b,#0f172a);
    border:2px solid #3b82f6; border-radius:14px; padding:24px;
    text-align:center; box-shadow:0 8px 20px rgba(59,130,246,.15);
}
.result-strength { font-size:3rem; font-weight:800; color:#60a5fa; margin:10px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: compat wrapper to silence use_container_width deprecation
# ─────────────────────────────────────────────────────────────────────────────
_st_ver = tuple(int(x) for x in st.__version__.split(".")[:2])

def _chart(fig, height=None):
    """Plotly chart – compatible with Streamlit ≥1.40 and ≥1.64."""
    if height:
        fig.update_layout(height=height)
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)

def _df(df_in):
    """Dataframe – compatible across versions."""
    try:
        st.dataframe(df_in, width="stretch")
    except Exception:
        st.dataframe(df_in, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ANN loader
# ─────────────────────────────────────────────────────────────────────────────
def create_and_load_ann():
    if os.path.exists('ann_weights.joblib'):
        try:
            return joblib.load('ann_weights.joblib')
        except Exception:
            pass
    if keras is not None and os.path.exists('ann_model.keras'):
        model = keras.Sequential([
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(128, activation='relu', name='dense_10'),
            keras.layers.Dense(64,  activation='relu', name='dense_11'),
            keras.layers.Dropout(0.2,                  name='dropout_2'),
            keras.layers.Dense(32,  activation='relu', name='dense_12'),
            keras.layers.Dense(16,  activation='relu', name='dense_13'),
            keras.layers.Dense(1,   activation='linear',name='dense_14'),
        ])
        try:
            model.load_weights('ann_model.keras')
        except Exception:
            pass
        return model
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    m = {}
    for name, fname in [
        ('XGBoost',           'xgboost.joblib'),
        ('Random Forest',     'random_forest.joblib'),
        ('Gradient Boosting', 'gradient_boosting.joblib'),
        ('SVR',               'svr.joblib'),
        ('Linear Regression', 'linear_regression.joblib'),
    ]:
        if os.path.exists(fname):
            m[name] = joblib.load(fname)
    m['Artificial Neural Network'] = create_and_load_ann()
    scaler = joblib.load('ann_scaler.joblib') if os.path.exists('ann_scaler.joblib') else None
    return m, scaler

@st.cache_data
def load_datasets():
    d = {}
    for key, fname in [
        ('comparison',        'model_comparison.csv'),
        ('feature_importance','feature_importance.csv'),
        ('cv',                '10_fold_cross_validation.csv'),
        ('test_pred',         'test_predictions.csv'),
    ]:
        if os.path.exists(fname):
            d[key] = pd.read_csv(fname)
    return d

models, ann_scaler = load_all_models()
datasets = load_datasets()

FEATURES = ['Cement','Blast Furnace Slag','Fly Ash','Water',
            'Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']

# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ann_forward(X_scaled):
    ann = models.get('Artificial Neural Network')
    if isinstance(ann, list):               # NumPy weights
        out = X_scaled
        for W, b in ann[:-1]:
            out = np.maximum(0, out @ W + b)
        W, b = ann[-1]
        return (out @ W + b).flatten()
    if ann is not None and keras is not None:
        return ann.predict(X_scaled, verbose=0).flatten()
    return np.zeros(len(X_scaled))

def predict(df_in, model_name):
    """Vectorised prediction; always returns ndarray."""
    sc = ann_scaler.transform(df_in) if ann_scaler is not None else df_in.values
    if model_name == 'Hybrid XGBoost + ANN':
        return (models['XGBoost'].predict(df_in) + _ann_forward(sc)) / 2
    if model_name == 'Artificial Neural Network':
        return _ann_forward(sc)
    return models[model_name].predict(df_in)

def predict_one(row_df, model_name):
    return float(predict(row_df, model_name)[0])

def category(s):
    if s < 20:  return "Low Strength Concrete",                "badge-warning", "Non-structural / walkways / curb stones."
    if s < 40:  return "Standard Structural Concrete",         "badge-std",     "Slabs, columns, beams, footings, reinforced walls."
    if s < 60:  return "High-Strength Concrete (HSC)",         "badge-high",    "High-rise pillars, pre-stressed girders, heavy bridges."
    return      "Ultra-High Performance Concrete (UHPC)",      "badge-high",    "Nuclear shielding, extreme-load marine structures, skyscrapers."

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – plain ASCII keys (no emoji) to avoid Linux encoding issues
# ─────────────────────────────────────────────────────────────────────────────
try:
    st.sidebar.image("https://img.icons8.com/isometric/96/brick.png", width=64)
except Exception:
    pass

st.sidebar.title("Navigation")

PAGES = {
    "Predictor":   "🧪 Interactive Predictor",
    "Models":      "📊 Model Comparison",
    "Features":    "🔍 Feature Analysis",
    "Dataset":     "📁 Dataset Explorer",
    "Specs":       "ℹ️  Research Specs",
}

page_key = st.sidebar.radio(
    "Go to:",
    list(PAGES.keys()),
    format_func=lambda k: PAGES[k],
    key="page_selector",
)

st.sidebar.markdown("---")
st.sidebar.caption("🔬 ML & ANN Modeling for Concrete Strength Prediction")
st.sidebar.caption("✨ Top Model: XGBoost  R² = 0.941")

ALL_MODELS = [
    "XGBoost","Hybrid XGBoost + ANN","Random Forest",
    "Gradient Boosting","Artificial Neural Network","SVR","Linear Regression",
]

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — INTERACTIVE PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
if page_key == "Predictor":
    st.markdown("""
    <div class="hero-banner">
      <h1 class="hero-title">🏗️ Concrete Compressive Strength Predictor</h1>
      <p class="hero-subtitle">Simulate concrete mix formulations and predict compressive strength (MPa).</p>
    </div>""", unsafe_allow_html=True)

    # Session state defaults
    defaults = dict(mix_cement=280.0, mix_slag=70.0, mix_flyash=50.0,
                    mix_water=180.0,  mix_sp=6.0,    mix_ca=980.0,
                    mix_fa=770.0,     mix_age=28)
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def _preset(c,s,f,w,sp,ca,fa,a):
        st.session_state.mix_cement  = float(c)
        st.session_state.mix_slag    = float(s)
        st.session_state.mix_flyash  = float(f)
        st.session_state.mix_water   = float(w)
        st.session_state.mix_sp      = float(sp)
        st.session_state.mix_ca      = float(ca)
        st.session_state.mix_fa      = float(fa)
        st.session_state.mix_age     = int(a)

    col_in, col_out = st.columns([1.6, 1.1])

    with col_in:
        st.subheader("⚙️ Mix Parameters")
        st.caption("Quick presets:")
        p0,p1,p2,p3 = st.columns(4)
        p0.button("Standard 28D", on_click=_preset, args=(280,70,50,180,6,980,770,28))
        p1.button("High-Strength", on_click=_preset, args=(450,100,0,150,12,950,720,28))
        p2.button("Eco Fly-Ash",  on_click=_preset, args=(200,0,160,165,8,1000,790,56))
        p3.button("Early 7-Day",  on_click=_preset, args=(380,120,0,175,9,920,750,7))

        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            cement  = st.number_input("Cement (kg/m³)",             100.0,540.0,step=5.0,  key="mix_cement")
            slag    = st.number_input("Blast Furnace Slag (kg/m³)",    0.0,360.0,step=5.0,  key="mix_slag")
            flyash  = st.number_input("Fly Ash (kg/m³)",               0.0,200.0,step=5.0,  key="mix_flyash")
            water   = st.number_input("Water (kg/m³)",               120.0,250.0,step=2.0,  key="mix_water")
        with c2:
            sp      = st.number_input("Superplasticizer (kg/m³)",      0.0, 35.0,step=0.5,  key="mix_sp")
            ca      = st.number_input("Coarse Aggregate (kg/m³)",    800.0,1150.0,step=10.0, key="mix_ca")
            fa      = st.number_input("Fine Aggregate (kg/m³)",      590.0, 950.0,step=10.0, key="mix_fa")
            age     = st.slider("Curing Age (Days)", 1, 365,                                  key="mix_age")

        binder = cement+slag+flyash
        wb = water/binder if binder>0 else 0
        dens = binder+water+sp+ca+fa
        st.markdown("##### 🧮 Derived Proportions")
        m1,m2,m3 = st.columns(3)
        m1.metric("w/b Ratio",  f"{wb:.3f}")
        m2.metric("Total Binder", f"{binder:.1f} kg/m³")
        m3.metric("Density",    f"{dens:.1f} kg/m³")

    with col_out:
        st.subheader("🎯 Prediction Output")
        sel = st.selectbox("Model:", ALL_MODELS, key="p1_model")
        row = pd.DataFrame([[cement,slag,flyash,water,sp,ca,fa,age]], columns=FEATURES)
        strength = predict_one(row, sel)
        cat_title, badge, usage = category(strength)

        st.markdown(f"""
        <div class="result-box">
          <div style="color:#94a3b8;font-size:.95rem;font-weight:600;">PREDICTED STRENGTH</div>
          <div class="result-strength">{strength:.2f} <span style="font-size:1.4rem;">MPa</span></div>
          <div style="margin-top:10px"><span class="{badge}">{cat_title}</span></div>
        </div>""", unsafe_allow_html=True)

        st.info(f"💡 **Use case**: {usage}")
        st.markdown("---")
        st.subheader("📊 All-Model Comparison")

        preds = {m: predict_one(row, m) for m in ALL_MODELS}
        pdf = pd.DataFrame(preds.items(), columns=["Model","Strength (MPa)"])
        fig = px.bar(pdf, x="Strength (MPa)", y="Model", orientation='h',
                     color="Strength (MPa)", color_continuous_scale='Blues',
                     text_auto='.2f')
        fig.update_layout(height=300, margin=dict(l=0,r=10,t=10,b=10),
                          showlegend=False, yaxis_title=None)
        _chart(fig)

    st.markdown("---")
    with st.expander("📁 Batch Prediction (Upload CSV)"):
        tmpl = pd.DataFrame([[280,70,50,180,6,980,770,28],[450,100,0,150,12,950,720,28]],
                            columns=FEATURES)
        st.download_button("📥 Download template", tmpl.to_csv(index=False),
                           "template.csv","text/csv")
        up = st.file_uploader("Upload CSV", type=["csv"], key="batch_up")
        if up:
            bdf = pd.read_csv(up)
            miss = [c for c in FEATURES if c not in bdf.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                res = bdf.copy()
                for m in ALL_MODELS:
                    res[f"{m} (MPa)"] = np.round(predict(bdf[FEATURES], m), 2)
                _df(res.head(10))
                st.download_button("📥 Download results", res.to_csv(index=False),
                                   "batch_results.csv","text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page_key == "Models":
    st.markdown("""
    <div class="hero-banner">
      <h1 class="hero-title">📊 Model Comparison & Benchmarks</h1>
      <p class="hero-subtitle">MAE, RMSE, R² and 10-Fold Cross-Validation across all models.</p>
    </div>""", unsafe_allow_html=True)

    comp = datasets.get('comparison', pd.DataFrame())
    cv   = datasets.get('cv',         pd.DataFrame())
    tpdf = datasets.get('test_pred',  pd.DataFrame())

    k1,k2,k3,k4 = st.columns(4)
    for col, lbl, val, sub in [
        (k1,"Best R²",      "0.941","XGBoost"),
        (k2,"Lowest MAE",   "2.61 MPa","XGBoost"),
        (k3,"Lowest RMSE",  "4.20 MPa","XGBoost"),
        (k4,"Hybrid R²",    "0.923","XGBoost+ANN"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{lbl}</div>
          <div class="metric-value">{val}</div>
          <div style="color:#94a3b8;font-size:.8rem;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard","🔄 10-Fold CV","📈 Scatter Plot"])

    with tab1:
        if not comp.empty:
            c1,c2 = st.columns([1.2,1])
            with c1:
                _df(comp)
            with c2:
                fig = px.bar(comp, x='R2', y='Model', orientation='h',
                             title='R² Score (Higher is Better)',
                             color='R2', color_continuous_scale='Viridis',
                             text_auto='.3f')
                fig.update_layout(showlegend=False,
                                  yaxis={'categoryorder':'total ascending'})
                _chart(fig, 320)
            fig2 = px.bar(comp, x='Model', y=['MAE','RMSE'], barmode='group',
                          title='MAE & RMSE (Lower is Better)',
                          color_discrete_sequence=['#38bdf8','#f43f5e'])
            fig2.update_layout(yaxis_title="Error (MPa)")
            _chart(fig2, 350)

    with tab2:
        if not cv.empty:
            _df(cv)
            fig = go.Figure(go.Bar(
                x=cv['Model'], y=cv['CV R2 Mean'],
                error_y=dict(type='data', array=cv['CV R2 Std'], visible=True),
                marker_color='#6366f1'))
            fig.update_layout(title="10-Fold CV R² ± Std", yaxis_title="Mean R²")
            _chart(fig, 380)

    with tab3:
        if not tpdf.empty:
            fig = px.scatter(tpdf, x='Actual Strength', y='Hybrid Predicted Strength',
                             hover_data=['Cement','Water','Age'],
                             color='Absolute Error', color_continuous_scale='Plasma',
                             title='Actual vs Hybrid Predicted Strength')
            lo = min(tpdf['Actual Strength'].min(), tpdf['Hybrid Predicted Strength'].min())
            hi = max(tpdf['Actual Strength'].max(), tpdf['Hybrid Predicted Strength'].max())
            fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='y=x',
                                     line=dict(color='#10b981',dash='dash',width=2)))
            _chart(fig, 480)

            fig2 = px.histogram(tpdf, x='Absolute Error', nbins=30,
                                title='Residual Distribution',
                                color_discrete_sequence=['#3b82f6'])
            _chart(fig2, 320)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif page_key == "Features":
    st.markdown("""
    <div class="hero-banner">
      <h1 class="hero-title">🔍 Feature Importance & Sensitivity</h1>
      <p class="hero-subtitle">How each mix parameter drives compressive strength.</p>
    </div>""", unsafe_allow_html=True)

    feat = datasets.get('feature_importance', pd.DataFrame())
    c1, c2 = st.columns([1,1.3])

    with c1:
        st.subheader("📌 Feature Importance")
        if not feat.empty:
            fsort = feat.sort_values('Importance', ascending=True)
            fig = px.bar(fsort, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues',
                         text_auto='.3f')
            fig.update_layout(showlegend=False, yaxis_title=None,
                              xaxis_title="Importance Score")
            _chart(fig, 450)
            st.info("💡 Age + Cement explain **66%** of strength variance.")

    with c2:
        st.subheader("📈 Strength Growth Simulator")
        sim_c  = st.slider("Cement (kg/m³)",  150, 500, 300, key="sc")
        sim_w  = st.slider("Water (kg/m³)",   130, 220, 180, key="sw")
        sim_sp = st.slider("Superplasticizer", 0.0, 20.0, 6.0, key="ssp")
        sim_m  = st.selectbox("Model:", ["XGBoost","Hybrid XGBoost + ANN",
                                          "Random Forest","Artificial Neural Network"],
                               key="sm")
        ages = np.arange(1, 181, 2)
        bdf  = pd.DataFrame([[sim_c,70,50,sim_w,sim_sp,950,750,a] for a in ages],
                             columns=FEATURES)
        y = predict(bdf, sim_m)
        fig = px.line(pd.DataFrame({'Age':ages,'Strength':y}),
                      x='Age', y='Strength',
                      title=f"Growth Curve – {sim_m}", markers=True)
        fig.update_traces(line_color='#38bdf8', line_width=3)
        fig.add_vline(x=28, line_dash='dash', line_color='#10b981',
                      annotation_text='28-Day')
        fig.update_layout(yaxis_title="Strength (MPa)")
        _chart(fig, 380)

    st.markdown("---")
    st.subheader("🌊 Water/Cement Ratio Sensitivity")
    wrange = np.linspace(130, 240, 30)
    wbdf   = pd.DataFrame([[300,70,50,w,6,950,750,28] for w in wrange],
                           columns=FEATURES)
    wy = predict(wbdf, "XGBoost")
    fig = px.line(pd.DataFrame({'w/c': wrange/300, 'Strength':wy}),
                  x='w/c', y='Strength',
                  title='Strength vs w/c Ratio – XGBoost', markers=True)
    fig.update_traces(line_color='#f43f5e', line_width=3)
    _chart(fig, 380)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATASET EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page_key == "Dataset":
    st.markdown("""
    <div class="hero-banner">
      <h1 class="hero-title">📁 Dataset Explorer & Downloads</h1>
      <p class="hero-subtitle">Filter and download test predictions, CV results, and model metrics.</p>
    </div>""", unsafe_allow_html=True)

    tpdf = datasets.get('test_pred', pd.DataFrame())
    if not tpdf.empty:
        st.subheader("📋 Test Predictions (203 samples)")
        fc1, fc2 = st.columns(2)
        with fc1:
            age_opts   = sorted(tpdf['Age'].unique().tolist())
            age_filter = st.multiselect("Curing Age:", age_opts, default=[7,28,90],
                                        key="d_age")
        with fc2:
            smin = float(tpdf['Actual Strength'].min())
            smax = float(tpdf['Actual Strength'].max())
            lo, hi = st.slider("Actual Strength (MPa):", smin, smax,
                               (10.0, 80.0), key="d_str")

        mask = (tpdf['Actual Strength'] >= lo) & (tpdf['Actual Strength'] <= hi)
        if age_filter:
            mask &= tpdf['Age'].isin(age_filter)
        _df(tpdf[mask])

        st.markdown("##### Downloads")
        d1,d2,d3 = st.columns(3)
        d1.download_button("Test Predictions CSV",
                           tpdf.to_csv(index=False),
                           "test_predictions.csv","text/csv",
                           key="dl_tp")
        comp = datasets.get('comparison', pd.DataFrame())
        d2.download_button("Model Comparison CSV",
                           comp.to_csv(index=False) if not comp.empty else "",
                           "model_comparison.csv","text/csv",
                           key="dl_mc")
        cv = datasets.get('cv', pd.DataFrame())
        d3.download_button("10-Fold CV CSV",
                           cv.to_csv(index=False) if not cv.empty else "",
                           "10_fold_cv.csv","text/csv",
                           key="dl_cv")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RESEARCH SPECS (default / else)
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="hero-banner">
      <h1 class="hero-title">ℹ️ Research Architecture & Specs</h1>
      <p class="hero-subtitle">ANN architecture, ensemble strategy, and methodology details.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns([1.2,1])
    with c1:
        st.subheader("🧠 ANN Architecture")
        st.code("""
Input  : 8 Features
Dense  : 128 units, ReLU
Dense  :  64 units, ReLU
Dropout: rate = 0.2
Dense  :  32 units, ReLU
Dense  :  16 units, ReLU
Output :   1 unit,  Linear  →  MPa
        """)
        st.markdown("""
- **Scaler**: `StandardScaler` (fitted on training data)  
- **Optimizer**: Adam (lr = 0.001)  
- **Loss**: Mean Squared Error  
        """)

    with c2:
        st.subheader("⚡ Hybrid Ensemble")
        st.markdown(r"""
$$\hat{y}_{hybrid} = 0.5 \cdot \hat{y}_{XGBoost} + 0.5 \cdot \hat{y}_{ANN}$$

| Model | R² |
|---|---|
| XGBoost | **0.941** |
| Hybrid  | **0.923** |
| GBM     | 0.915 |
| RF      | 0.910 |
| ANN     | 0.875 |
| SVR     | 0.880 |
| LR      | 0.580 |
        """)

    st.markdown("---")
    st.caption("Concrete Compressive Strength ML & ANN Research | Streamlit · XGBoost · Scikit-Learn")
