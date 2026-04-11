"""
ACR Dashboard - Agentic Counterfactual Reasoning Web Application
A domain-agnostic tool for generating and auditing counterfactual explanations.
Features AUTOMATIC causal rule detection — no manual configuration needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acr.engine import ACREngine
from acr.smart_rules import auto_detect_rules, apply_rules
from acr.narrator import get_narrative, generate_explanation, evaluate_explanation
from acr.faithfulness_metrics import create_evaluator

# ---- AUTO-CLEANING FUNCTION ----
def auto_clean_dataframe(df, clean_mode='auto'):
    """Professor-approved auto-cleaning for messy CSVs"""
    df_clean = df.copy()
    
    print(f"🔍 Original: {len(df)} rows, {len(df.columns)} cols")
    
    if clean_mode == 'raw':
        print("⚠️ Using RAW data (no cleaning)")
        return df_clean, {"status": "raw", "changes": {}}
    
    cleaning_log = {"status": "cleaned", "changes": {}}
    
    # 1. REMOVE DUPLICATES
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dup_removed = initial_rows - len(df_clean)
    if dup_removed > 0:
        print(f"🧹 Removed {dup_removed} duplicates")
        cleaning_log["changes"]["duplicates_removed"] = dup_removed
    
    # 2. HANDLE MISSING VALUES
    missing_before = df_clean.isnull().sum().sum()
    for col in df_clean.columns:
        if df_clean[col].dtype in ['object', 'string']:
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
            else:
                df_clean[col] = df_clean[col].fillna('unknown')
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    if missing_before > 0:
        print(f"🧹 Filled {missing_before} missing values")
        cleaning_log["changes"]["missing_filled"] = missing_before
    
    # 3. FIX DATA TYPES
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            temp_col = pd.to_numeric(df_clean[col], errors='coerce')
            if temp_col.notna().sum() / len(temp_col) > 0.8:
                df_clean[col] = temp_col
            else:
                try:
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                except Exception as e:
                    print(f"⚠️ Could not encode {col}: {e}")
    
    # 4. REMOVE OUTLIERS (IQR method)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            if outlier_count > 0:
                print(f"🧹 Removed {outlier_count} outliers from {col}")
                if "outliers_removed" not in cleaning_log["changes"]:
                    cleaning_log["changes"]["outliers_removed"] = 0
                cleaning_log["changes"]["outliers_removed"] += outlier_count
    
    print(f"✅ CLEANED: {len(df_clean)} rows, {len(df_clean.columns)} cols")
    cleaning_log["changes"]["final_rows"] = len(df_clean)
    return df_clean, cleaning_log

# ---- Page Configuration ----
st.set_page_config(
    page_title="ACR Dashboard - Counterfactual Explanations",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p { font-size: 1rem; opacity: 0.9; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border: 1px solid #e0e4f5;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    .metric-card h3 { font-size: 2rem; font-weight: 700; color: #667eea; margin: 0; }
    .metric-card p { font-size: 0.85rem; color: #6b7280; margin: 0.25rem 0 0 0; }

    .rule-immutable {
        background: #fee2e2; border-left: 4px solid #ef4444;
        padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.4rem;
    }
    .rule-constraint {
        background: #fef3c7; border-left: 4px solid #f59e0b;
        padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.4rem;
    }
    .rule-mutable {
        background: #dcfce7; border-left: 4px solid #22c55e;
        padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 0.4rem;
    }

    .step-box {
        background: #ffffff; border: 2px solid #e5e7eb; border-radius: 12px;
        padding: 1.2rem; margin-bottom: 0.75rem; transition: all 0.3s ease;
    }
    .step-box.active { border-color: #667eea; background: #f8f9ff; box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15); }
    .step-box.done { border-color: #22c55e; background: #f0fdf4; }
    .step-number {
        display: inline-block; width: 28px; height: 28px; border-radius: 50%;
        background: #667eea; color: white; text-align: center; line-height: 28px;
        font-weight: 700; font-size: 0.85rem; margin-right: 0.5rem;
    }

    .faithful-row { background-color: #dcfce7 !important; }
    .faithless-row { background-color: #fef2f2 !important; }

    .result-header {
        font-size: 1.3rem; font-weight: 600; color: #1f2937;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #667eea;
    }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%); }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] p { color: #e0e7ff !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---- Helper Functions ----
def safe_dataframe_for_streamlit(df):
    """
    Convert dataframe to be safe for Streamlit display by ensuring all columns are strings.
    This prevents PyArrow serialization errors with mixed data types.
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    # Convert all columns to strings to avoid PyArrow issues
    for col in df_copy.columns:
        try:
            # Handle different data types safely
            if df_copy[col].dtype == 'object':
                # For object columns, convert to string but handle None/NaN
                df_copy[col] = df_copy[col].fillna('').astype(str)
            else:
                # For numeric/bool columns, convert to string representation
                df_copy[col] = df_copy[col].astype(str)
        except Exception:
            # Fallback: force everything to string
            df_copy[col] = df_copy[col].fillna('').astype(str)
    
    return df_copy

# ---- Session State ----
defaults = {
    'engine': ACREngine(), 'step': 1, 'model_trained': False,
    'cfs_generated': False, 'audit_done': False, 'query_dict': None,
    'raw_cfs': [], 'valid_cfs': [], 'invalid_cfs': [], 'auto_rules': {},
    'narrative': None, 'sample_choice': None, 'faithfulness_metrics': None,
    'clean_mode': 'auto', 'cleaning_log': {}, 'dataset_name': 'uploaded dataset'
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🔮 ACR Dashboard</h1>
    <p>Agentic Counterfactual Reasoning — Upload any dataset, generate explanations, auto-audit for faithfulness</p>
</div>
""", unsafe_allow_html=True)


# ---- Sidebar ----
with st.sidebar:
    st.markdown("## 🧭 Pipeline Steps")
    steps = [
        ("Upload Dataset", st.session_state.step > 1),
        ("Train Model", st.session_state.model_trained),
        ("Generate & Auto-Audit", st.session_state.audit_done),
        ("Batch Analysis", hasattr(st.session_state, 'batch_results') and st.session_state.get('batch_results')),
    ]
    for i, (label, done) in enumerate(steps, 1):
        status = "done" if done else ("active" if i == st.session_state.step else "")
        icon = "✅" if done else ("🔵" if i == st.session_state.step else "⚪")
        st.markdown(f'<div class="step-box {status}"><span class="step-number">{i}</span> {icon} {label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 How It Works")
    st.markdown("""
    1. **Upload** any dataset (CSV/Excel/JSON)
    2. **Train** a classifier on your data
    3. **Generate** counterfactual suggestions
    4. **Auto-Audit**: The system **automatically** detects which features are immutable (age, race, genetics) and filters impossible suggestions — **no manual setup needed!**
    5. **Batch Analysis**: Run faithful/faithless classification on all instances
    """)


# ═══════════════════════════════════════
# STEP 1: UPLOAD
# ═══════════════════════════════════════
st.markdown('<div class="result-header">📁 Step 1: Upload Your Dataset</div>', unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 2])

with col_upload:
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=['csv', 'xlsx', 'xls', 'json'], key="uploader")
    st.markdown("**Or try a sample:**")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("🩺 Diabetes", use_container_width=True):
            st.session_state.sample_choice = 'diabetes'
            st.session_state.model_trained = False
            st.session_state.cfs_generated = False
            st.session_state.audit_done = False
    with sc2:
        if st.button("💰 Adult Income", use_container_width=True):
            st.session_state.sample_choice = 'adult'
            st.session_state.model_trained = False
            st.session_state.cfs_generated = False
            st.session_state.audit_done = False

    # 🧹 AUTO-CLEAN TOGGLE
    st.markdown("---")
    st.markdown("**🧹 Data Cleaning Option:**")
    col_toggle1, col_toggle2 = st.columns([1, 2])
    with col_toggle1:
        clean_checkbox = st.checkbox("Auto-clean data", value=True, key="clean_toggle")
        st.session_state.clean_mode = 'auto' if clean_checkbox else 'raw'
    with col_toggle2:
        if clean_checkbox:
            st.markdown("<span style='color: #22c55e; font-weight: bold;'>✅ Auto-clean: Removes duplicates, fills missing, fixes types & outliers</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color: #ef4444; font-weight: bold;'>❌ Raw data: Analyze as-is (no cleaning)</span>", unsafe_allow_html=True)

engine = st.session_state.engine
df = None

if uploaded_file:
    try:
        df = engine.load_data(uploaded_file)
        st.session_state.dataset_name = uploaded_file.name
        # Apply auto-cleaning
        df, cleaning_log = auto_clean_dataframe(df, st.session_state.clean_mode)
        st.session_state.cleaning_log = cleaning_log
        st.session_state.step = max(st.session_state.step, 2)
    except Exception as e:
        st.error(f"Error: {e}")
elif hasattr(st.session_state, 'sample_choice') and st.session_state.sample_choice:
    try:
        if st.session_state.sample_choice == 'diabetes':
            engine.df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
            st.session_state.dataset_name = "Diabetes (Sample Dataset)"
        elif st.session_state.sample_choice == 'adult':
            cols = ['age','workclass','fnlwgt','education','education_num','marital_status',
                    'occupation','relationship','race','sex','capital_gain','capital_loss',
                    'hours_per_week','native_country','income']
            engine.df = pd.read_csv(
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
                header=None, names=cols, skipinitialspace=True
            ).head(2000)
            st.session_state.dataset_name = "Adult Income (Sample Dataset)"
        engine.df.columns = [c.strip().replace(' ', '_') for c in engine.df.columns]
        df = engine.df
        # Apply auto-cleaning
        df, cleaning_log = auto_clean_dataframe(df, st.session_state.clean_mode)
        st.session_state.cleaning_log = cleaning_log
        st.session_state.step = max(st.session_state.step, 2)
    except Exception as e:
        st.error(f"Error loading sample: {e}")

if df is not None:
    with col_preview:
        st.markdown(f"**Loaded:** `{df.shape[0]}` rows × `{df.shape[1]}` columns")
        preview_df = df.head(8)
        st.dataframe(safe_dataframe_for_streamlit(preview_df), use_container_width=True, height=280)

    # 🧹 SHOW CLEANING STATUS
    if st.session_state.cleaning_log.get("status") == "cleaned":
        with st.expander("🧹 Data Cleaning Summary", expanded=False):
            changes = st.session_state.cleaning_log.get("changes", {})
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if "duplicates_removed" in changes:
                    st.metric("🗑️ Duplicates Removed", changes["duplicates_removed"])
                if "missing_filled" in changes:
                    st.metric("📊 Missing Values Filled", changes["missing_filled"])
            with col_c2:
                if "outliers_removed" in changes:
                    st.metric("📈 Outliers Removed", changes["outliers_removed"])
                if "final_rows" in changes:
                    st.metric("✅ Final Rows", changes["final_rows"])
    elif st.session_state.cleaning_log.get("status") == "raw":
        st.info("⚠️ Using RAW data (no cleaning applied)", icon="⚠️")

    # 🧹 SHOW CLEANING STATUS
    if st.session_state.cleaning_log.get("status") == "cleaned":
        with st.expander("🧹 Data Cleaning Summary", expanded=False):
            changes = st.session_state.cleaning_log.get("changes", {})
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if "duplicates_removed" in changes:
                    st.metric("🗑️ Duplicates Removed", changes["duplicates_removed"])
                if "missing_filled" in changes:
                    st.metric("📊 Missing Values Filled", changes["missing_filled"])
            with col_c2:
                if "outliers_removed" in changes:
                    st.metric("📈 Outliers Removed", changes["outliers_removed"])
                if "final_rows" in changes:
                    st.metric("✅ Final Rows", changes["final_rows"])
    elif st.session_state.cleaning_log.get("status") == "raw":
        st.info("⚠️ Using RAW data (no cleaning applied)", icon="⚠️")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3><p>Total Rows</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>{len(df.select_dtypes(include=[np.number]).columns)}</h3><p>Numerical</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><h3>{len(df.select_dtypes(include=["object"]).columns)}</h3><p>Categorical</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════
    # STEP 2: TRAIN MODEL
    # ═══════════════════════════════════════
    st.markdown('<div class="result-header">🎯 Step 2: Configure & Train Model</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("🎯 Select Target Feature", options=df.columns.tolist(), index=len(df.columns)-1)
    with c2:
        st.markdown(f"**Target `{target_col}` distribution:**")
        st.write(df[target_col].value_counts().head(10))

    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training RandomForest classifier..."):
            try:
                engine.detect_features(target_col)
                accuracy = engine.train_model()
                st.session_state.model_trained = True
                st.session_state.cfs_generated = False
                st.session_state.audit_done = False
                st.session_state.step = max(st.session_state.step, 3)
                st.success(f"✅ Model trained! Accuracy: **{accuracy:.1%}**")
                
                # Display faithfulness metrics if available
                if hasattr(st.session_state, 'faithfulness_metrics') and st.session_state.faithfulness_metrics:
                    metrics = st.session_state.faithfulness_metrics
                    st.markdown("---")
                    st.markdown('<div class="result-header">📊 Faithfulness Metrics</div>', unsafe_allow_html=True)
                    
                    # Faithfulness metrics cards
                    fm1, fm2, fm3 = st.columns(3)
                    with fm1:
                        faithful_rate = (metrics['num_faithful_cf'] / max(metrics['num_counterfactuals'], 1)) * 100
                        st.markdown(f'<div class="metric-card"><h3 style="color:#22c55e">{faithful_rate:.1f}%</h3><p>✅ Faithful Rate<br><small>Actionable + Improving</small></p></div>', unsafe_allow_html=True)
                    with fm2:
                        rule_violation_rate = ((metrics['num_actionable_cf'] - metrics['num_faithful_cf']) / max(metrics['num_counterfactuals'], 1)) * 100
                        st.markdown(f'<div class="metric-card"><h3 style="color:#ef4444">{rule_violation_rate:.1f}%</h3><p>❌ Rule Violations<br><small>Invalid Changes</small></p></div>', unsafe_allow_html=True)
                    with fm3:
                        agent_full_rate = 100.0 if metrics['has_feasible_recourse'] else 0.0
                        st.markdown(f'<div class="metric-card"><h3 style="color:#f59e0b">{agent_full_rate:.1f}%</h3><p>🎯 Agent Full Rate<br><small>Feasible Recourse</small></p></div>', unsafe_allow_html=True)
                    
                    # Additional metrics
                    st.markdown("**Detailed Metrics:**")
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    with detail_col1:
                        st.metric("Total Counterfactuals", metrics['num_counterfactuals'])
                        st.metric("Actionable CFs", metrics['num_actionable_cf'])
                    with detail_col2:
                        st.metric("Faithful CFs", metrics['num_faithful_cf'])
                        st.metric("Improving CFs", metrics['num_improving_cf'])
                    with detail_col3:
                        st.metric("Avg Feasibility", f"{metrics['avg_feasibility_score']:.2f}")
                        st.metric("Max Improvement", f"{metrics['max_improvement_delta']:.3f}")
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ═══════════════════════════════════════
    # STEP 3: GENERATE + AUTO-AUDIT
    # ═══════════════════════════════════════
    if st.session_state.model_trained:
        st.markdown("---")
        st.markdown('<div class="result-header">🔮 Step 3: Generate Counterfactuals & Auto-Audit</div>', unsafe_allow_html=True)

        # Auto-detect rules and show them
        auto_rules = auto_detect_rules(engine.feature_names, df)
        st.session_state.auto_rules = auto_rules

        st.markdown("#### 🧠 Auto-Detected Causal Rules")
        st.markdown("*The system automatically identified these constraints from your column names:*")

        immutable_feats = [f for f, r in auto_rules.items() if not r['mutable']]
        constrained_feats = [f for f, r in auto_rules.items() if r['mutable'] and r['constraint']]
        mutable_feats = [f for f, r in auto_rules.items() if r['mutable'] and not r['constraint']]

        rule_col1, rule_col2, rule_col3 = st.columns(3)
        with rule_col1:
            st.markdown(f"**🔒 Immutable ({len(immutable_feats)})**")
            for f in immutable_feats:
                st.markdown(f'<div class="rule-immutable">🚫 <strong>{f}</strong><br><small>{auto_rules[f]["reason"]}</small></div>', unsafe_allow_html=True)
            if not immutable_feats:
                st.info("None detected")
        with rule_col2:
            st.markdown(f"**⚠️ Constrained ({len(constrained_feats)})**")
            for f in constrained_feats:
                st.markdown(f'<div class="rule-constraint">⬆️ <strong>{f}</strong><br><small>{auto_rules[f]["reason"]}</small></div>', unsafe_allow_html=True)
            if not constrained_feats:
                st.info("None detected")
        with rule_col3:
            st.markdown(f"**✅ Mutable ({len(mutable_feats)})**")
            for f in mutable_feats:
                st.markdown(f'<div class="rule-mutable">✏️ <strong>{f}</strong><br><small>Can be changed freely</small></div>', unsafe_allow_html=True)

        st.markdown("---")

        # CF generation controls
        cf_c1, cf_c2, cf_c3 = st.columns(3)
        with cf_c1:
            test_samples = engine.get_test_samples(20)
            query_idx = st.selectbox("📋 Select test sample", range(len(test_samples)), format_func=lambda i: f"Sample {i+1}")
        with cf_c2:
            predicted = engine.get_predicted_class(query_idx)
            target_classes = engine.get_target_classes()
            st.markdown(f"**Current prediction:** `{predicted}`")
            desired = st.selectbox("🎯 Desired outcome", options=target_classes, index=0)
        with cf_c3:
            num_cfs = st.slider("Number of counterfactuals", 3, 10, 5)

        st.markdown("**Selected Instance:**")
        instance_df = test_samples.iloc[[query_idx]]
        st.dataframe(safe_dataframe_for_streamlit(instance_df), use_container_width=True)

        if st.button("⚡ Generate & Auto-Audit", type="primary", use_container_width=True):
            with st.spinner("Generating counterfactuals and running causal audit..."):
                try:
                    desired_enc = desired
                    if engine.target in engine.label_encoders:
                        desired_enc = engine.label_encoders[engine.target].transform([str(desired)])[0]

                    query_dict, raw_cfs = engine.generate_counterfactuals(query_idx, desired_enc, num_cfs)
                    st.session_state.query_dict = query_dict
                    st.session_state.raw_cfs = raw_cfs

                    # AUTO-AUDIT using smart rules
                    valid, invalid = apply_rules(query_dict, raw_cfs, auto_rules)
                    st.session_state.valid_cfs = valid
                    st.session_state.invalid_cfs = invalid
                    st.session_state.cfs_generated = True
                    st.session_state.audit_done = True
                    st.session_state.narrative = None  # Reset for fresh LLM call

                    # Compute faithfulness metrics
                    evaluator = create_evaluator(auto_rules)
                    
                    # Get original prediction probability (not class label)
                    original_instance = engine.X_test.iloc[[query_idx]]
                    original_pred_probs = engine.model.predict_proba(original_instance)[0]
                    original_pred_prob = float(original_pred_probs[desired_enc])
                    
                    instance_metrics = evaluator.compute_instance_faithfulness(
                        instance_id=query_idx,
                        original_features=pd.Series(query_dict),
                        counterfactuals=[pd.Series(cf) for cf in raw_cfs],
                        original_prediction=original_pred_prob,
                        model=engine.model,
                        desired_class=desired_enc
                    )
                    st.session_state.faithfulness_metrics = instance_metrics

                    st.success(f"✅ Generated **{len(raw_cfs)}** suggestions → **{len(valid)}** Faithful, **{len(invalid)}** Faithless")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ═══════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════
    if st.session_state.audit_done:
        st.markdown("---")

        # ---- LLM Narrative (FRESH GENERATION PER INSTANCE) ----
        st.markdown('<div class="result-header">🤖 AI-Generated Explanation (Instance-Specific)</div>', unsafe_allow_html=True)
        
        # ALWAYS generate fresh narrative for each instance (no caching)
        try:
            with st.spinner("🧠 Generating personalized explanation..."):
                # Get model prediction for context
                model_pred = engine.model.predict_proba(engine.X_test.iloc[[query_idx]])[0]
                pred_label = "Favorable" if model_pred.argmax() == 1 else "Unfavorable"
                pred_prob = model_pred.max()
                
                narrative = generate_explanation(
                    instance_id=query_idx,
                    query_dict=st.session_state.query_dict,
                    valid_cfs=st.session_state.valid_cfs,
                    invalid_cfs=st.session_state.invalid_cfs,
                    feature_names=engine.feature_names,
                    model_pred=f"{pred_label} ({pred_prob:.1%})",
                    dataset_name=st.session_state.get('dataset_name', 'uploaded dataset')
                )
                
                # Evaluate explanation quality
                is_valid_explanation, quality_score = evaluate_explanation(narrative, st.session_state.valid_cfs, st.session_state.invalid_cfs)
                
                if not is_valid_explanation:
                    print(f"[DEBUG] WARNING: Low-quality explanation (score: {quality_score:.2f})")
                    st.warning(f"⚠️ Explanation quality: {quality_score:.0%} (May be auto-generated due to API limits)")
                
        except Exception as e:
            narrative = f"Could not generate explanation: {e}"
            print(f"[DEBUG] Exception: {e}")
            st.error(narrative)
        
        st.info(narrative, icon="🤖")

        st.markdown("---")
        st.markdown('<div class="result-header">📊 Audit Results — Faithful vs Faithless</div>', unsafe_allow_html=True)

        total = len(st.session_state.raw_cfs)
        n_valid = len(st.session_state.valid_cfs)
        n_invalid = len(st.session_state.invalid_cfs)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="metric-card"><h3>{total}</h3><p>Total Generated</p></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card"><h3 style="color:#16a34a">{n_valid}</h3><p>✅ Faithful (Actionable)</p></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card"><h3 style="color:#dc2626">{n_invalid}</h3><p>❌ Faithless (Discarded)</p></div>', unsafe_allow_html=True)

        # Donut chart
        if total > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Faithful ✅', 'Faithless ❌'], values=[n_valid, n_invalid],
                marker=dict(colors=['#22c55e', '#ef4444']), hole=0.5,
                textinfo='label+value', textfont=dict(size=14)
            )])
            fig_pie.update_layout(title="Audit Summary", height=350, margin=dict(t=50, b=20, l=20, r=20), font=dict(family="Inter"))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Valid CFs
        if st.session_state.valid_cfs:
            st.markdown("### ✅ Faithful Suggestions (Actionable)")
            for i, cf in enumerate(st.session_state.valid_cfs, 1):
                with st.expander(f"✅ Suggestion #{i}", expanded=(i == 1)):
                    changes = []
                    for feat in engine.feature_names:
                        orig = st.session_state.query_dict.get(feat)
                        new = cf.get(feat)
                        if str(orig) != str(new):
                            try:
                                diff = float(new) - float(orig)
                                direction = "📈" if diff > 0 else "📉"
                                changes.append({"Feature": feat, "Original": str(orig), "Suggested": str(new), "Change": f"{direction} {diff:+.2f}"})
                            except (ValueError, TypeError):
                                changes.append({"Feature": feat, "Original": str(orig), "Suggested": str(new), "Change": "🔄 Changed"})
                    if changes:
                        changes_df = pd.DataFrame(changes)
                        st.dataframe(safe_dataframe_for_streamlit(changes_df), use_container_width=True, hide_index=True)

        # Invalid CFs
        if st.session_state.invalid_cfs:
            st.markdown("### ❌ Faithless Suggestions (Auto-Discarded)")
            for i, item in enumerate(st.session_state.invalid_cfs, 1):
                st.markdown(f"""
                <div class="rule-immutable">
                    <strong>Discarded #{i}:</strong> {item['reason']}
                </div>
                """, unsafe_allow_html=True)

        # Comparison bar chart
        if st.session_state.valid_cfs:
            st.markdown("### 📈 Feature Comparison: Original vs Best Suggestion")
            best_cf = st.session_state.valid_cfs[0]
            num_feats, orig_v, cf_v = [], [], []
            for f in engine.feature_names:
                try:
                    o = float(st.session_state.query_dict.get(f, 0))
                    c = float(best_cf.get(f, 0))
                    num_feats.append(f)
                    orig_v.append(o)
                    cf_v.append(c)
                except (ValueError, TypeError):
                    continue
            if num_feats:
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Original', x=num_feats, y=orig_v, marker_color='#6366f1', text=[f'{v:.1f}' for v in orig_v], textposition='auto'))
                fig.add_trace(go.Bar(name='Counterfactual', x=num_feats, y=cf_v, marker_color='#22c55e', text=[f'{v:.1f}' for v in cf_v], textposition='auto'))
                fig.update_layout(barmode='group', height=400, title="Numerical Feature Changes", font=dict(family="Inter"))
                st.plotly_chart(fig, use_container_width=True)

        # Export
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        export = {
            "original_instance": st.session_state.query_dict,
            "auto_detected_rules": {f: {"mutable": r["mutable"], "constraint": r["constraint"], "reason": r["reason"]} for f, r in st.session_state.auto_rules.items()},
            "total_generated": total, "faithful_count": n_valid, "faithless_count": n_invalid,
            "faithful_suggestions": st.session_state.valid_cfs,
            "faithless_suggestions": st.session_state.invalid_cfs,
        }
        st.download_button("📥 Download Full Audit Report (JSON)", json.dumps(export, indent=4, default=str), "acr_audit_report.json", "application/json", use_container_width=True)

# ═══════════════════════════════════════
# BATCH ANALYSIS SECTION
# ═══════════════════════════════════════

# Run batch analysis on uploaded data
if st.session_state.model_trained and df is not None:
    st.markdown("---")
    st.markdown("### 🚀 Run Batch Analysis on Your Data")

    if st.button("🔍 Analyze All Instances (Batch)", type="secondary", use_container_width=True):
        with st.spinner("Running batch analysis on all instances..."):
            try:
                # Get test samples (limit to reasonable number for demo)
                test_samples = engine.get_test_samples(min(20, len(df)))
                batch_results = []

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, instance_idx in enumerate(range(len(test_samples))):
                    status_text.text(f"Analyzing instance {i+1}/{len(test_samples)}...")

                    # Generate CFs for this instance
                    desired_class = engine.get_target_classes()[0]  # Use first target class
                    desired_enc = desired_class
                    if engine.target in engine.label_encoders:
                        desired_enc = engine.label_encoders[engine.target].transform([str(desired_class)])[0]

                    query_dict, raw_cfs = engine.generate_counterfactuals(instance_idx, desired_enc, 5)

                    # Apply rules
                    
                    valid, invalid = apply_rules(query_dict, raw_cfs, st.session_state.auto_rules)

                    # Get original prediction probability (not class label)
                    original_instance = engine.X_test.iloc[[instance_idx]]
                    original_pred_probs = engine.model.predict_proba(original_instance)[0]
                    original_pred_prob = float(original_pred_probs[desired_enc])

                    # Compute metrics
                    evaluator = create_evaluator(st.session_state.auto_rules)
                    metrics = evaluator.compute_instance_faithfulness(
                        instance_id=instance_idx,
                        original_features=pd.Series(query_dict),
                        counterfactuals=[pd.Series(cf) for cf in raw_cfs],
                        original_prediction=original_pred_prob,
                        model=engine.model,
                        desired_class=desired_enc
                    )

                    batch_results.append(metrics)
                    progress_bar.progress((i + 1) / len(test_samples))

                progress_bar.empty()
                status_text.empty()

                # Display batch results
                st.success(f"✅ Batch analysis completed for {len(batch_results)} instances!")

                # Convert to DataFrame
                batch_df = pd.DataFrame(batch_results)

                # Overall batch metrics
                batch_total_cfs = batch_df['num_counterfactuals'].sum()
                batch_total_faithful = batch_df['num_faithful_cf'].sum()
                batch_total_actionable = batch_df['num_actionable_cf'].sum()
                batch_agent_full = batch_df['has_feasible_recourse'].sum()

                batch_faithful_rate = (batch_total_faithful / max(batch_total_cfs, 1)) * 100
                batch_violation_rate = ((batch_total_actionable - batch_total_faithful) / max(batch_total_cfs, 1)) * 100
                batch_agent_rate = (batch_agent_full / max(len(batch_df), 1)) * 100

                st.markdown("#### 📊 Your Dataset Batch Results")
                br1, br2, br3 = st.columns(3)
                with br1:
                    st.markdown(f'<div class="metric-card"><h3 style="color:#22c55e">{batch_faithful_rate:.1f}%</h3><p>✅ Faithful Rate</p></div>', unsafe_allow_html=True)
                with br2:
                    st.markdown(f'<div class="metric-card"><h3 style="color:#ef4444">{batch_violation_rate:.1f}%</h3><p>❌ Rule Violations</p></div>', unsafe_allow_html=True)
                with br3:
                    st.markdown(f'<div class="metric-card"><h3 style="color:#f59e0b">{batch_agent_rate:.1f}%</h3><p>🎯 Agent Full Rate</p></div>', unsafe_allow_html=True)

                # Show per-instance results
                st.markdown("#### 📋 Per-Instance Results")
                instance_results = []
                for _, row in batch_df.iterrows():
                    instance_id = int(row['instance_id'])
                    num_faithful = int(row['num_faithful_cf'])
                    num_cfs = int(row['num_counterfactuals'])
                    has_recourse = row['has_feasible_recourse']

                    if num_faithful > 0:
                        status = "✅ FAITHFUL"
                    else:
                        status = "❌ FAITHLESS"

                    instance_results.append({
                        'Instance': f"#{instance_id}",
                        'Status': status,
                        'Faithful CFs': num_faithful,
                        'Total CFs': num_cfs,
                        'Agent-Full': "Yes" if has_recourse else "No"
                    })

                results_df = pd.DataFrame(instance_results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Download batch results CSV
                st.download_button(
                    "📥 Download Batch Results (CSV)",
                    batch_df.to_csv(index=False),
                    "batch_faithfulness_results.csv",
                    "text/csv",
                    use_container_width=True
                )

                # Download total batch audit report JSON
                batch_export = {
                    "batch_summary": {
                        "total_instances": len(batch_df),
                        "total_counterfactuals": int(batch_total_cfs),
                        "total_faithful_cf": int(batch_total_faithful),
                        "total_actionable_cf": int(batch_total_actionable),
                        "agent_full_instances": int(batch_agent_full),
                        "faithful_rate": float(batch_faithful_rate),
                        "rule_violation_rate": float(batch_violation_rate),
                        "agent_full_rate": float(batch_agent_rate)
                    },
                    "instances": batch_df.to_dict(orient='records')
                }
                st.download_button(
                    "📥 Download Total Batch Report (JSON)",
                    json.dumps(batch_export, indent=4, default=str),
                    "batch_total_report.json",
                    "application/json",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"Batch analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# Footer
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.85rem; padding:1rem;">
    <strong>ACR Dashboard</strong> — Agentic Counterfactual Reasoning |
    Built for XAI Project (6th Semester) |
    Powered by DiCE, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
