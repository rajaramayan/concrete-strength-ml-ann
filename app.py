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

# Set page config
st.set_page_config(
    page_title="Concrete Strength ML & ANN Research Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual design
st.markdown("""
<style>
    /* Global Styles & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
    }
    
    /* Card Container */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 600;
    }
    
    /* Highlight Badge */
    .badge-high {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-std {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #3b82f6;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.15);
    }
    .result-strength {
        font-size: 3rem;
        font-weight: 800;
        color: #60a5fa;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load ANN weights / model
def create_and_load_ann():
    if os.path.exists('ann_weights.joblib'):
        return joblib.load('ann_weights.joblib')
    if keras is not None:
        model = keras.Sequential([
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(128, activation='relu', name='dense_10'),
            keras.layers.Dense(64, activation='relu', name='dense_11'),
            keras.layers.Dropout(0.2, name='dropout_2'),
            keras.layers.Dense(32, activation='relu', name='dense_12'),
            keras.layers.Dense(16, activation='relu', name='dense_13'),
            keras.layers.Dense(1, activation='linear', name='dense_14')
        ])
        if os.path.exists('ann_model.keras'):
            try:
                model.load_weights('ann_model.keras')
            except Exception:
                pass
        return model
    return None

# Cache resources
@st.cache_resource
def load_all_models():
    models = {}
    if os.path.exists('xgboost.joblib'):
        models['XGBoost'] = joblib.load('xgboost.joblib')
    if os.path.exists('random_forest.joblib'):
        models['Random Forest'] = joblib.load('random_forest.joblib')
    if os.path.exists('gradient_boosting.joblib'):
        models['Gradient Boosting'] = joblib.load('gradient_boosting.joblib')
    if os.path.exists('svr.joblib'):
        models['SVR'] = joblib.load('svr.joblib')
    if os.path.exists('linear_regression.joblib'):
        models['Linear Regression'] = joblib.load('linear_regression.joblib')
    
    models['Artificial Neural Network'] = create_and_load_ann()
    
    scaler = None
    if os.path.exists('ann_scaler.joblib'):
        scaler = joblib.load('ann_scaler.joblib')
        
    return models, scaler

@st.cache_data
def load_datasets():
    data = {}
    if os.path.exists('model_comparison.csv'):
        data['comparison'] = pd.read_csv('model_comparison.csv')
    if os.path.exists('feature_importance.csv'):
        data['feature_importance'] = pd.read_csv('feature_importance.csv')
    if os.path.exists('10_fold_cross_validation.csv'):
        data['cv'] = pd.read_csv('10_fold_cross_validation.csv')
    if os.path.exists('test_predictions.csv'):
        data['test_pred'] = pd.read_csv('test_predictions.csv')
    return data

models, ann_scaler = load_all_models()
datasets = load_datasets()

feature_names = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/96/brick.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section:",
    [
        "🧪 Interactive Predictor",
        "📊 Model Comparison & Benchmarks",
        "🔍 Feature Analysis & Sensitivity",
        "📁 Research Dataset Explorer",
        "ℹ️ Research Specs & Architecture"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🔬 **Research Focus**: ML & ANN Modeling for Concrete Strength Prediction")
st.sidebar.caption("✨ **Top Model**: XGBoost ($R^2 = 0.941$)")

# Pure NumPy ANN inference
def predict_ann(input_df):
    scaled = ann_scaler.transform(input_df) if ann_scaler else input_df.values
    ann_obj = models.get('Artificial Neural Network')
    if isinstance(ann_obj, list): # ann_weights list
        out = scaled
        for W, b in ann_obj[:-1]:
            out = np.maximum(0, np.dot(out, W) + b)
        W_last, b_last = ann_obj[-1]
        out = np.dot(out, W_last) + b_last
        return float(out[0, 0]) if out.ndim > 1 else float(out[0])
    elif ann_obj is not None and keras is not None:
        return float(ann_obj.predict(scaled, verbose=0)[0, 0])
    return 0.0

# Prediction Helper Function
def predict_strength(input_df, model_name):
    if model_name == "Hybrid XGBoost + ANN":
        xgb_p = float(models['XGBoost'].predict(input_df)[0])
        ann_p = predict_ann(input_df)
        return (xgb_p + ann_p) / 2.0
    elif model_name == "Artificial Neural Network":
        return predict_ann(input_df)
    else:
        return float(models[model_name].predict(input_df)[0])

def get_concrete_category(strength):
    if strength < 20:
        return "Low Strength Concrete", "badge-warning", "Non-structural, blinding concrete, walkways, curb stones."
    elif strength < 40:
        return "Standard Structural Concrete", "badge-std", "Residential slabs, columns, beams, footings, reinforced walls."
    elif strength < 60:
        return "High-Strength Concrete (HSC)", "badge-high", "High-rise structural pillars, pre-stressed girders, heavy bridges."
    else:
        return "Ultra-High Performance Concrete (UHPC)", "badge-high", "Specialized nuclear shielding, extreme load marine structures, skyscrapers."

# ----------------------------------------------------
# PAGE 1: INTERACTIVE PREDICTOR
# ----------------------------------------------------
if page == "🧪 Interactive Predictor":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🏗️ Concrete Compressive Strength Predictor</h1>
        <p class="hero-subtitle">Simulate concrete mix formulations and predict 28-day+ compressive strength (MPa) using Machine Learning & Deep Learning ANN ensembles.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_inputs, col_results = st.columns([1.6, 1.1])
    
    with col_inputs:
        st.subheader("⚙️ Concrete Mix Parameters")
        
        # Preset formulations
        st.caption("⚡ Quick Preset Mix Templates:")
        preset_cols = st.columns(4)
        
        # Default state initialization
        if 'cement' not in st.session_state:
            st.session_state.cement = 280.0
            st.session_state.slag = 70.0
            st.session_state.flyash = 50.0
            st.session_state.water = 180.0
            st.session_state.superplasticizer = 6.0
            st.session_state.coarse_agg = 980.0
            st.session_state.fine_agg = 770.0
            st.session_state.age = 28
            
        if preset_cols[0].button("Standard 28D"):
            st.session_state.cement, st.session_state.slag, st.session_state.flyash = 280.0, 70.0, 50.0
            st.session_state.water, st.session_state.superplasticizer = 180.0, 6.0
            st.session_state.coarse_agg, st.session_state.fine_agg = 980.0, 770.0
            st.session_state.age = 28
            st.rerun()
            
        if preset_cols[1].button("High-Strength"):
            st.session_state.cement, st.session_state.slag, st.session_state.flyash = 450.0, 100.0, 0.0
            st.session_state.water, st.session_state.superplasticizer = 150.0, 12.0
            st.session_state.coarse_agg, st.session_state.fine_agg = 950.0, 720.0
            st.session_state.age = 28
            st.rerun()
            
        if preset_cols[2].button("Eco Fly-Ash"):
            st.session_state.cement, st.session_state.slag, st.session_state.flyash = 200.0, 0.0, 160.0
            st.session_state.water, st.session_state.superplasticizer = 165.0, 8.0
            st.session_state.coarse_agg, st.session_state.fine_agg = 1000.0, 790.0
            st.session_state.age = 56
            st.rerun()

        if preset_cols[3].button("Early 7-Day"):
            st.session_state.cement, st.session_state.slag, st.session_state.flyash = 380.0, 120.0, 0.0
            st.session_state.water, st.session_state.superplasticizer = 175.0, 9.0
            st.session_state.coarse_agg, st.session_state.fine_agg = 920.0, 750.0
            st.session_state.age = 7
            st.rerun()

        st.markdown("---")
        
        i_col1, i_col2 = st.columns(2)
        with i_col1:
            cement = st.number_input("Cement (kg/m³)", 100.0, 540.0, value=st.session_state.cement, step=5.0)
            slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 360.0, value=st.session_state.slag, step=5.0)
            flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 200.0, value=st.session_state.flyash, step=5.0)
            water = st.number_input("Water (kg/m³)", 120.0, 250.0, value=st.session_state.water, step=2.0)
            
        with i_col2:
            superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 35.0, value=st.session_state.superplasticizer, step=0.5)
            coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 800.0, 1150.0, value=st.session_state.coarse_agg, step=10.0)
            fine_agg = st.number_input("Fine Aggregate (kg/m³)", 590.0, 950.0, value=st.session_state.fine_agg, step=10.0)
            age = st.slider("Curing Age (Days)", 1, 365, value=st.session_state.age)

        # Derived metrics calculation
        total_binder = cement + slag + flyash
        water_binder_ratio = water / total_binder if total_binder > 0 else 0
        total_density = total_binder + water + superplasticizer + coarse_agg + fine_agg
        
        st.markdown("##### 🧮 Derived Mix Proportions")
        m1, m2, m3 = st.columns(3)
        m1.metric("Water-to-Binder Ratio (w/b)", f"{water_binder_ratio:.3f}")
        m2.metric("Total Binder", f"{total_binder:.1f} kg/m³")
        m3.metric("Estimated Density", f"{total_density:.1f} kg/m³")

    with col_results:
        st.subheader("🎯 Prediction Output")
        
        model_options = [
            "XGBoost",
            "Hybrid XGBoost + ANN",
            "Random Forest",
            "Gradient Boosting",
            "Artificial Neural Network",
            "SVR",
            "Linear Regression"
        ]
        selected_model = st.selectbox("Select ML / DL Model:", model_options, index=0)
        
        input_data = pd.DataFrame([[cement, slag, flyash, water, superplasticizer, coarse_agg, fine_agg, age]],
                                  columns=feature_names)
        
        predicted_strength = predict_strength(input_data, selected_model)
        category_title, badge_class, usage_desc = get_concrete_category(predicted_strength)
        
        st.markdown(f"""
        <div class="result-box">
            <div style="color: #94a3b8; font-size: 0.95rem; font-weight: 600;">PREDICTED COMPRESSIVE STRENGTH</div>
            <div class="result-strength">{predicted_strength:.2f} <span style="font-size: 1.4rem;">MPa</span></div>
            <div style="margin-top: 10px;">
                <span class="{badge_class}">{category_title}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info(f"💡 **Recommended Application**: {usage_desc}")
        
        st.markdown("---")
        st.subheader("📊 Multi-Model Predictions Comparison")
        
        # Calculate prediction across all available models
        all_preds = {}
        for m in model_options:
            all_preds[m] = predict_strength(input_data, m)
            
        pred_df = pd.DataFrame(list(all_preds.items()), columns=['Model', 'Predicted Strength (MPa)'])
        pred_df['Difference vs Selected (MPa)'] = pred_df['Predicted Strength (MPa)'] - predicted_strength
        
        fig_comp = px.bar(
            pred_df, 
            x='Predicted Strength (MPa)', 
            y='Model', 
            orientation='h',
            color='Predicted Strength (MPa)',
            color_continuous_scale='Blues',
            text_auto='.2f'
        )
        fig_comp.update_layout(
            height=300,
            margin=dict(l=0, r=10, t=10, b=10),
            xaxis_title="Strength (MPa)",
            yaxis_title=None,
            showlegend=False
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Batch Prediction Section
    st.markdown("---")
    with st.expander("📁 Batch Prediction Tool (Upload CSV)", expanded=False):
        st.write("Upload a CSV file containing concrete mix parameters to run batch predictions across all research models.")
        sample_download = pd.DataFrame([
            [280.0, 70.0, 50.0, 180.0, 6.0, 980.0, 770.0, 28],
            [450.0, 100.0, 0.0, 150.0, 12.0, 950.0, 720.0, 28]
        ], columns=feature_names)
        
        st.download_button("📥 Download Sample CSV Template", sample_download.to_csv(index=False), "sample_concrete_mixes.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Upload Concrete Data CSV", type=['csv'])
        if uploaded_file:
            user_batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in feature_names if c not in user_batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                st.success(f"Successfully loaded {len(user_batch_df)} rows for batch prediction!")
                
                results_df = user_batch_df.copy()
                for m in model_options:
                    preds_list = []
                    for idx, row in user_batch_df[feature_names].iterrows():
                        row_df = pd.DataFrame([row.values], columns=feature_names)
                        preds_list.append(predict_strength(row_df, m))
                    results_df[f"Pred_{m} (MPa)"] = preds_list
                    
                st.dataframe(results_df.head(10), use_container_width=True)
                
                csv_out = results_df.to_csv(index=False)
                st.download_button("📥 Download Batch Predictions CSV", csv_out, "concrete_batch_predictions.csv", "text/csv")

# ----------------------------------------------------
# PAGE 2: MODEL COMPARISON & BENCHMARKS
# ----------------------------------------------------
elif page == "📊 Model Comparison & Benchmarks":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📊 Model Comparison & Performance Benchmarks</h1>
        <p class="hero-subtitle">Comprehensive evaluation metrics (MAE, RMSE, R²) and 10-Fold Cross-Validation analysis for machine learning & ANN models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    comp_df = datasets.get('comparison', pd.DataFrame())
    cv_df = datasets.get('cv', pd.DataFrame())
    test_pred_df = datasets.get('test_pred', pd.DataFrame())
    
    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Model R²</div>
            <div class="metric-value">0.941</div>
            <div style="color: #94a3b8; font-size: 0.8rem;">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Lowest MAE</div>
            <div class="metric-value">2.61 <span style="font-size:1rem;">MPa</span></div>
            <div style="color: #94a3b8; font-size: 0.8rem;">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Lowest RMSE</div>
            <div class="metric-value">4.20 <span style="font-size:1rem;">MPa</span></div>
            <div style="color: #94a3b8; font-size: 0.8rem;">XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Hybrid XGB+ANN R²</div>
            <div class="metric-value">0.923</div>
            <div style="color: #94a3b8; font-size: 0.8rem;">Ensemble Blend</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏆 Performance Leaderboard", "🔄 10-Fold Cross Validation", "📈 Actual vs Predicted Scatter"])
    
    with tab1:
        st.subheader("Model Evaluation Summary")
        if not comp_df.empty:
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(
                    comp_df.style.highlight_max(axis=0, subset=['R2'], color='rgba(16,185,129,0.3)')
                           .highlight_min(axis=0, subset=['MAE', 'RMSE'], color='rgba(16,185,129,0.3)'),
                    use_container_width=True
                )
            with c2:
                fig_r2 = px.bar(
                    comp_df,
                    x='R2',
                    y='Model',
                    orientation='h',
                    title='R² Score Comparison (Higher is Better)',
                    color='R2',
                    color_continuous_scale='Viridis',
                    text_auto='.3f'
                )
                fig_r2.update_layout(height=320, showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_r2, use_container_width=True)
                
            fig_err = px.bar(
                comp_df,
                x='Model',
                y=['MAE', 'RMSE'],
                barmode='group',
                title='MAE and RMSE Error Comparison (Lower is Better)',
                color_discrete_sequence=['#38bdf8', '#f43f5e']
            )
            fig_err.update_layout(height=350, yaxis_title="Error (MPa)")
            st.plotly_chart(fig_err, use_container_width=True)
            
    with tab2:
        st.subheader("10-Fold Cross-Validation Metrics")
        if not cv_df.empty:
            st.write("Cross-validation ensures model generalization and evaluates performance stability across 10 distinct dataset splits.")
            st.dataframe(cv_df, use_container_width=True)
            
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Bar(
                x=cv_df['Model'],
                y=cv_df['CV R2 Mean'],
                error_y=dict(type='data', array=cv_df['CV R2 Std'], visible=True),
                marker_color='#6366f1',
                name='CV R² Mean ± Std'
            ))
            fig_cv.update_layout(
                title="10-Fold CV R² Score with Standard Deviation Error Bars",
                yaxis_title="Mean R² Score",
                height=380
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    with tab3:
        st.subheader("Actual vs Predicted Compressive Strength Scatter Plot")
        if not test_pred_df.empty:
            st.write("Evaluating test predictions against ground truth actual compressive strength values.")
            
            fig_scat = px.scatter(
                test_pred_df,
                x='Actual Strength',
                y='Hybrid Predicted Strength',
                hover_data=['Cement', 'Water', 'Age'],
                title='Actual Strength vs Hybrid (XGBoost + ANN) Predicted Strength',
                labels={'Actual Strength': 'Actual Strength (MPa)', 'Hybrid Predicted Strength': 'Predicted Strength (MPa)'},
                color='Absolute Error',
                color_continuous_scale='Plasma'
            )
            
            # Identity line y = x
            min_val = min(test_pred_df['Actual Strength'].min(), test_pred_df['Hybrid Predicted Strength'].min())
            max_val = max(test_pred_df['Actual Strength'].max(), test_pred_df['Hybrid Predicted Strength'].max())
            fig_scat.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Fit Line (y=x)',
                line=dict(color='#10b981', dash='dash', width=2)
            ))
            fig_scat.update_layout(height=480)
            st.plotly_chart(fig_scat, use_container_width=True)
            
            # Residual Distribution
            fig_hist = px.histogram(
                test_pred_df,
                x='Absolute Error',
                nbins=30,
                title='Absolute Prediction Error Distribution (Residuals)',
                color_discrete_sequence=['#3b82f6']
            )
            fig_hist.update_layout(height=320, xaxis_title="Absolute Error (MPa)", yaxis_title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)

# ----------------------------------------------------
# PAGE 3: FEATURE ANALYSIS & SENSITIVITY
# ----------------------------------------------------
elif page == "🔍 Feature Analysis & Sensitivity":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🔍 Feature Importance & Mix Sensitivity Simulator</h1>
        <p class="hero-subtitle">Gain deep explainable AI insights into how concrete mix ingredients and curing age influence structural compressive strength.</p>
    </div>
    """, unsafe_allow_html=True)
    
    feat_df = datasets.get('feature_importance', pd.DataFrame())
    
    col_feat, col_sens = st.columns([1, 1.3])
    
    with col_feat:
        st.subheader("📌 Feature Importance Ranking")
        if not feat_df.empty:
            feat_df_sorted = feat_df.sort_values(by='Importance', ascending=True)
            fig_feat = px.bar(
                feat_df_sorted,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues',
                text_auto='.3f'
            )
            fig_feat.update_layout(
                height=450,
                xaxis_title="Relative Feature Importance Score",
                yaxis_title=None,
                showlegend=False
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            st.info("💡 **Key Insight**: Curing **Age** and **Cement content** account for over **66%** of total model feature importance!")

    with col_sens:
        st.subheader("📈 Dynamic Curing Age Strength Growth Simulator")
        st.write("Simulate how compressive strength develops over time (1 to 365 Days) for a baseline concrete mix.")
        
        sim_c = st.slider("Cement (kg/m³)", 150, 500, 300, key="sim_c")
        sim_w = st.slider("Water (kg/m³)", 130, 220, 180, key="sim_w")
        sim_sp = st.slider("Superplasticizer (kg/m³)", 0.0, 20.0, 6.0, key="sim_sp")
        
        sim_model = st.selectbox("Simulation Model:", ["XGBoost", "Hybrid XGBoost + ANN", "Random Forest", "Artificial Neural Network"])
        
        ages_range = np.arange(1, 181, 2)
        sim_results = []
        for a in ages_range:
            sim_input = pd.DataFrame([[sim_c, 70.0, 50.0, sim_w, sim_sp, 950.0, 750.0, a]], columns=feature_names)
            sim_results.append(predict_strength(sim_input, sim_model))
            
        sim_curve_df = pd.DataFrame({'Age (Days)': ages_range, 'Predicted Strength (MPa)': sim_results})
        
        fig_curve = px.line(
            sim_curve_df,
            x='Age (Days)',
            y='Predicted Strength (MPa)',
            title=f'Concrete Compressive Strength Growth Curve ({sim_model})',
            markers=True
        )
        fig_curve.update_traces(line_color='#38bdf8', line_width=3)
        fig_curve.add_vline(x=28, line_dash="dash", line_color="#10b981", annotation_text="28-Day Standard Benchmark", annotation_position="top left")
        fig_curve.update_layout(height=380)
        st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown("---")
    st.subheader("🌊 Water-to-Cement Ratio ($w/c$) Sensitivity Plot")
    st.write("Holding all other ingredients fixed, evaluate how altering water content impacts compressive strength.")
    
    water_range = np.linspace(130, 240, 30)
    wc_results = []
    for w in water_range:
        sim_input = pd.DataFrame([[300.0, 70.0, 50.0, w, 6.0, 950.0, 750.0, 28]], columns=feature_names)
        wc_results.append(predict_strength(sim_input, "XGBoost"))
        
    wc_df = pd.DataFrame({'Water Content (kg/m³)': water_range, 'Water/Cement Ratio': water_range / 300.0, 'Predicted Strength (MPa)': wc_results})
    
    fig_wc = px.line(
        wc_df,
        x='Water/Cement Ratio',
        y='Predicted Strength (MPa)',
        title='Strength vs Water-to-Cement Ratio (w/c) - XGBoost Model',
        markers=True
    )
    fig_wc.update_traces(line_color='#f43f5e', line_width=3)
    fig_wc.update_layout(height=380)
    st.plotly_chart(fig_wc, use_container_width=True)

# ----------------------------------------------------
# PAGE 4: RESEARCH DATASET EXPLORER
# ----------------------------------------------------
elif page == "📁 Research Dataset Explorer":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">📁 Research Dataset Explorer & Download Center</h1>
        <p class="hero-subtitle">Inspect test set predictions, cross-validation records, and research experimental data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    test_pred_df = datasets.get('test_pred', pd.DataFrame())
    
    if not test_pred_df.empty:
        st.subheader("📋 Test Predictions Dataset (203 Samples)")
        
        st.markdown("##### Filter Data by Curing Age & Strength:")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            age_filter = st.multiselect("Select Curing Age (Days):", sorted(test_pred_df['Age'].unique().tolist()), default=[7, 28, 90])
        with f_col2:
            min_str, max_str = st.slider("Filter Actual Strength (MPa):", 
                                         float(test_pred_df['Actual Strength'].min()), 
                                         float(test_pred_df['Actual Strength'].max()), 
                                         (10.0, 80.0))
            
        filtered_df = test_pred_df[
            (test_pred_df['Age'].isin(age_filter) if age_filter else True) &
            (test_pred_df['Actual Strength'] >= min_str) &
            (test_pred_df['Actual Strength'] <= max_str)
        ]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        st.markdown("##### 📥 Export Data Files:")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.download_button(
                "📥 Download Test Predictions CSV", 
                test_pred_df.to_csv(index=False), 
                "test_predictions.csv", 
                "text/csv"
            )
        with d_col2:
            comp_df = datasets.get('comparison', pd.DataFrame())
            st.download_button(
                "📥 Download Model Comparison CSV", 
                comp_df.to_csv(index=False) if not comp_df.empty else "", 
                "model_comparison.csv", 
                "text/csv"
            )
        with d_col3:
            cv_df = datasets.get('cv', pd.DataFrame())
            st.download_button(
                "📥 Download 10-Fold CV CSV", 
                cv_df.to_csv(index=False) if not cv_df.empty else "", 
                "10_fold_cross_validation.csv", 
                "text/csv"
            )

# ----------------------------------------------------
# PAGE 5: RESEARCH SPECS & ARCHITECTURE
# ----------------------------------------------------
elif page == "ℹ️ Research Specs & Architecture":
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">ℹ️ Research Architecture & Technical Specifications</h1>
        <p class="hero-subtitle">Detailed documentation of the Artificial Neural Network (ANN) architecture, ensemble strategies, and data pipeline.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c_arch1, c_arch2 = st.columns([1.2, 1])
    
    with c_arch1:
        st.subheader("🧠 Artificial Neural Network (ANN) Deep Architecture")
        st.markdown("""
        The Deep Learning model is constructed using a multi-layer perceptron (MLP) architecture with non-linear activation functions and dropout regularization to prevent overfitting on concrete mix feature distributions.
        
        ```
        Input Layer: 8 Features (Cement, Slag, Fly Ash, Water, SP, Coarse Agg, Fine Agg, Age)
             │
             ▼
        Dense Layer 1: 128 Units (ReLU Activation)
             │
             ▼
        Dense Layer 2: 64 Units (ReLU Activation)
             │
             ▼
        Dropout Layer: 0.2 Rate (Regularization)
             │
             ▼
        Dense Layer 3: 32 Units (ReLU Activation)
             │
             ▼
        Dense Layer 4: 16 Units (ReLU Activation)
             │
             ▼
        Output Layer: 1 Unit (Linear Activation -> Compressive Strength MPa)
        ```
        
        - **Preprocessing**: Input features are standardized using `StandardScaler` fitted on training data.
        - **Optimizer**: Adam Optimizer (learning rate = 0.001).
        - **Loss Function**: Mean Squared Error (MSE).
        """)
        
    with c_arch2:
        st.subheader("⚡ Hybrid XGBoost + ANN Ensemble Strategy")
        st.markdown("""
        The research features a **Hybrid Stacking/Blending Ensemble** that combines gradient boosting decision trees (XGBoost) with deep neural network feature representations (ANN).
        
        **Ensemble Formulation:**
        $$\\hat{y}_{hybrid} = 0.5 \\cdot \\hat{y}_{XGBoost} + 0.5 \\cdot \\hat{y}_{ANN}$$
        
        **Advantages:**
        1. **Complementary Strengths**: XGBoost captures non-linear tabular interactions and decision boundaries, while ANN models smooth continuous non-linear response surfaces.
        2. **Robustness**: Reduces variance and sensitivity to outlier mix proportions.
        3. **R² Score**: Achieves an exceptional **0.923 $R^2$** on out-of-sample test datasets.
        """)

    st.markdown("---")
    st.caption("🔬 Concrete Compressive Strength ML & ANN Research Project | Built with Streamlit, TensorFlow, XGBoost & Scikit-Learn.")
