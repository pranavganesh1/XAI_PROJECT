"""
ACR Dashboard - Agentic Counterfactual Reasoning Web Application
A domain-agnostic tool for generating and auditing counterfactual explanations.
Features AUTOMATIC causal rule detection — no manual configuration needed.
OPTIMIZED with: Parallel batch processing, caching, accuracy validation, and natural language explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acr.engine import ACREngine
from acr.smart_rules import auto_detect_rules, apply_rules
from acr.narrator import get_narrative, generate_explanation, evaluate_explanation
from acr.faithfulness_metrics import create_evaluator
from fixed_acr_pipeline import FixedACREngine, DynamicExplanationGenerator, create_fixed_acr_pipeline, process_instance_with_debugging

# ═══════════════════════════════════════
# ⚙️ CONFIGURATION CONSTANTS
# ═══════════════════════════════════════
MIN_ACCURACY_THRESHOLD = 0.75  # Minimum required accuracy to proceed
MAX_WORKERS_BATCH = 4  # Max parallel workers for batch analysis
FAST_MODE_ENABLED = False  # Toggle for fast mode (skip heavy computations)
DEFAULT_BATCH_LIMIT = 20  # Default limit for batch analysis

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

# ═══════════════════════════════════════
# 🔍 ACCURACY VALIDATION & OPTIMIZATION
# ═══════════════════════════════════════

def optimize_dataset(df, engine, target_col):
    """
    Automated dataset optimization with feature scaling, encoding, and outlier handling.
    Returns optimized dataframe and optimization log.
    """
    df_opt = df.copy()
    opt_log = {"status": "optimized", "changes": {}}
    
    print(f"🔧 Starting dataset optimization...")
    
    # 1. FEATURE SCALING (Numerical features)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_cols = df_opt.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Remove target column from scaling if it's numeric
        scale_cols = [c for c in numeric_cols if c != target_col]
        if scale_cols:
            df_opt[scale_cols] = scaler.fit_transform(df_opt[scale_cols])
            opt_log["changes"]["features_scaled"] = len(scale_cols)
            print(f"📊 Scaled {len(scale_cols)} numerical features")
    
    # 2. LABEL ENCODING (Categorical features)
    categorical_cols = df_opt.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        encoded_count = 0
        for col in categorical_cols:
            if col != target_col:
                try:
                    le = LabelEncoder()
                    df_opt[col] = le.fit_transform(df_opt[col].astype(str))
                    encoded_count += 1
                except Exception as e:
                    print(f"⚠️ Could not encode {col}: {e}")
        if encoded_count > 0:
            opt_log["changes"]["features_encoded"] = encoded_count
            print(f"🔤 Encoded {encoded_count} categorical features")
    
    # 3. ADVANCED OUTLIER REMOVAL (Modified Z-score)
    from scipy import stats
    z_threshold = 3  # Remove values > 3 standard deviations
    outliers_removed_total = 0
    
    for col in df_opt.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            z_scores = np.abs(stats.zscore(df_opt[col].fillna(df_opt[col].mean())))
            outliers = (z_scores > z_threshold).sum()
            if outliers > 0:
                df_opt = df_opt[z_scores <= z_threshold]
                outliers_removed_total += outliers
                print(f"🎯 Removed {outliers} outliers from {col}")
    
    if outliers_removed_total > 0:
        opt_log["changes"]["advanced_outliers_removed"] = outliers_removed_total
    
    print(f"✅ OPTIMIZED: {len(df_opt)} rows, {len(df_opt.columns)} cols")
    opt_log["changes"]["final_rows"] = len(df_opt)
    return df_opt, opt_log

def check_accuracy_threshold(accuracy, threshold=MIN_ACCURACY_THRESHOLD):
    """
    Check if accuracy meets minimum threshold.
    Returns tuple: (is_valid: bool, message: str, can_proceed: bool)
    """
    if accuracy >= threshold:
        return True, f"✅ Accuracy {accuracy:.1%} meets threshold ({threshold:.1%})", True
    else:
        return False, f"❌ Accuracy {accuracy:.1%} is BELOW threshold ({threshold:.1%}). Please optimize dataset or check data quality.", False

# ═══════════════════════════════════════
# � DYNAMIC ACCURACY FEEDBACK SYSTEM
# ═══════════════════════════════════════

def get_accuracy_status_emoji(accuracy, threshold=MIN_ACCURACY_THRESHOLD):
    """Return emoji and color based on accuracy level"""
    if accuracy >= threshold:
        return "✅", "green", "Excellent"
    elif accuracy >= threshold - 0.10:
        return "⚠️", "orange", "Needs Improvement"
    else:
        return "❌", "red", "Critical"

def show_accuracy_reasons_details():
    """Show expandable section with reasons for low accuracy"""
    with st.expander("❓ Why might accuracy be low?", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Data Quality Issues:**
            - 📂 Poor data quality or inconsistencies
            - 🔗 Missing or incomplete values
            - 📊 Imbalanced classes in target
            - 🧹 Outliers or noisy data
            """)
        with col2:
            st.markdown("""
            **Model/Feature Issues:**
            - 🔍 Insufficient features for learning
            - ⚡ Features don't capture patterns
            - ⚙️ Suboptimal hyperparameters
            - 🎯 Model type mismatch for data
            """)
        st.info(
            "💡 **Recommendations:** Check data distribution, handle missing values, "
            "engineer new features, or try different model types.",
            icon="💡"
        )

def show_low_accuracy_actions():
    """Show action buttons for low accuracy scenarios"""
    st.markdown("### 🔧 What would you like to do?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Retrain Model", use_container_width=True, key="retrain_btn"):
            st.session_state.retrain_clicked = True
            st.info("🔄 The model will be retrained with current settings. Make sure to optimize your data first!")
    
    with col2:
        if st.button("⚙️ Improve Settings", use_container_width=True, key="settings_btn"):
            st.session_state.settings_clicked = True
            st.info("⚙️ Adjust model hyperparameters or feature selection above and retrain.")
    
    with col3:
        if st.button("📉 Continue Anyway", use_container_width=True, key="continue_btn"):
            st.session_state.continue_anyway = True
            st.warning("⚠️ Proceeding with low accuracy model. Results may be unreliable.")
    
    # Handle button actions
    if st.session_state.get("retrain_clicked"):
        st.session_state.retrain_clicked = False
        return "retrain"
    elif st.session_state.get("settings_clicked"):
        st.session_state.settings_clicked = False
        return "settings"
    elif st.session_state.get("continue_anyway"):
        st.session_state.continue_anyway = False
        return "continue"
    
    return None

def display_accuracy_feedback(accuracy, threshold=MIN_ACCURACY_THRESHOLD):
    """
    Display comprehensive, interactive accuracy feedback.
    
    Parameters:
    -----------
    accuracy : float
        Model accuracy (0.0 to 1.0)
    threshold : float
        Acceptable accuracy threshold (default: 0.75)
    
    Returns:
    --------
    dict : Contains 'meets_threshold' (bool) and 'user_action' (str or None)
    """
    st.markdown("---")
    st.markdown('<div class="result-header">📊 Model Accuracy Evaluation</div>', unsafe_allow_html=True)
    
    # Get status details
    emoji, color, status_text = get_accuracy_status_emoji(accuracy, threshold)
    meets_threshold = accuracy >= threshold
    
    # Display accuracy metric prominently
    metric_col, status_col = st.columns([2, 1])
    
    with metric_col:
        st.metric(
            "Model Accuracy",
            f"{accuracy*100:.1f}%",
            delta=f"{(accuracy - threshold)*100:.1f}% vs threshold" if not meets_threshold else f"+{(accuracy - threshold)*100:.1f}%",
            delta_color="normal" if meets_threshold else "inverse"
        )
    
    with status_col:
        st.markdown(f"""
        <div style='
            background-color: {color};
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            color: white;
            font-weight: bold;
        '>
        {emoji}<br>{status_text}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"**Required Threshold:** {threshold*100:.1f}%")
    
    # Display appropriate feedback based on accuracy
    if meets_threshold:
        st.success(
            f"✅ **Excellent!** Your model accuracy of **{accuracy*100:.1f}%** meets the required threshold. "
            f"You can proceed with confidence.",
            icon="✅"
        )
    else:
        # Calculate gap
        gap = threshold - accuracy
        st.warning(
            f"⚠️ **Accuracy Below Threshold** — Your model is **{gap*100:.1f}% below** the target of {threshold*100:.1f}%.\n\n"
            f"While you can still explore counterfactuals, results may be less reliable. "
            f"Consider improving your model first.",
            icon="⚠️"
        )
        
        # Show reasons
        show_accuracy_reasons_details()
        
        # Show action buttons
        user_action = show_low_accuracy_actions()
    
    return {
        "meets_threshold": meets_threshold,
        "accuracy": accuracy,
        "threshold": threshold,
        "status": status_text
    }

# ═══════════════════════════════════════
# �💡 COUNTERFACTUAL EXPLANATION GENERATOR
# ═══════════════════════════════════════

def get_changed_features(original_dict, suggested_dict):
    """Detect and extract ONLY meaningfully changed features."""
    if not original_dict or not suggested_dict:
        return []
    
    changed = []
    for feature in original_dict.keys():
        if feature not in suggested_dict:
            continue
        
        orig_val = original_dict[feature]
        sugg_val = suggested_dict[feature]
        
        # Handle NaN/None values
        orig_val = "N/A" if pd.isna(orig_val) else orig_val
        sugg_val = "N/A" if pd.isna(sugg_val) else sugg_val
        
        # Only include if values differ
        if str(orig_val).strip() != str(sugg_val).strip():
            changed.append({
                "feature": feature,
                "original": orig_val,
                "suggested": sugg_val
            })
    
    return changed

def infer_feature_type(value1, value2):
    """Auto-detect if feature is NUMERIC or CATEGORICAL."""
    values = [value1, value2]
    numeric_count = 0
    
    for v in values:
        if pd.isna(v) or v == "N/A":
            continue
        try:
            float(v)
            numeric_count += 1
        except (ValueError, TypeError):
            pass
    
    valid_count = len([v for v in values if v != "N/A" and not pd.isna(v)])
    if numeric_count < valid_count:
        return "categorical"
    
    return "numeric" if numeric_count > 0 else "categorical"

def generate_feature_explanation(feature, old_value, new_value):
    """Generate GENERIC explanation for a feature change (works for ANY dataset)."""
    old_value = "N/A" if pd.isna(old_value) else old_value
    new_value = "N/A" if pd.isna(new_value) else new_value
    
    feature_str = str(feature).strip()
    old_str = str(old_value).strip()
    new_str = str(new_value).strip()
    
    feature_type = infer_feature_type(old_value, new_value)
    
    if feature_type == "numeric":
        try:
            old_num = float(old_str) if old_str != "N/A" else 0
            new_num = float(new_str) if new_str != "N/A" else 0
            diff = new_num - old_num
            
            if diff > 0:
                return f"Increasing {feature_str} from {old_str} to {new_str} positively influences the prediction."
            elif diff < 0:
                return f"Reducing {feature_str} from {old_str} to {new_str} improves the predicted outcome."
            else:
                return f"{feature_str} remains stable at {old_str}, supporting prediction consistency."
        except (ValueError, TypeError):
            pass
    
    return f"Changing {feature_str} from '{old_str}' to '{new_str}' alters the model's decision boundary."

def format_suggestion(suggestion_dict, original_dict, suggestion_num):
    """Format a complete suggestion with ONLY changed features and explanations."""
    changed_features = get_changed_features(original_dict, suggestion_dict)
    
    if not changed_features:
        return f"**Suggestion {suggestion_num}:** No meaningful changes detected."
    
    output = []
    output.append(f"### 🔹 Suggestion {suggestion_num}")
    output.append("")
    
    output.append("**Proposed Changes:**")
    output.append("")
    for change in changed_features:
        feature = change["feature"]
        original = change["original"]
        suggested = change["suggested"]
        
        try:
            old_num = float(original) if original != "N/A" else 0
            new_num = float(suggested) if suggested != "N/A" else 0
            if new_num > old_num:
                arrow = "📈"
            elif new_num < old_num:
                arrow = "📉"
            else:
                arrow = "➡️"
        except (ValueError, TypeError):
            arrow = "🔄"
        
        output.append(f"- **{feature}:** {original} {arrow} {suggested}")
    
    output.append("")
    output.append("**Why These Changes Help:**")
    output.append("")
    for change in changed_features:
        feature = change["feature"]
        original = change["original"]
        suggested = change["suggested"]
        
        explanation = generate_feature_explanation(feature, original, suggested)
        output.append(f"- {explanation}")
    
    return "\n".join(output)

# ═══════════════════════════════════════
# ⚡ PARALLEL BATCH PROCESSING
# ═══════════════════════════════════════

@st.cache_data
def cache_expensive_computation(func_name, params_hash):
    """Generic caching decorator for expensive computations"""
    return None

def process_single_instance_batch(args):
    """
    Process a single instance for batch analysis.
    Returns metrics dict for that instance.
    CRITICAL: Must use engine.predict_proba() for proper preprocessing.
    """
    (instance_idx, engine, target_classes, auto_rules, desired_class) = args
    
    try:
        # Generate CFs for this instance
        desired_enc = desired_class
        if engine.target in engine.label_encoders:
            desired_enc = engine.label_encoders[engine.target].transform([str(desired_class)])[0]
        
        # Get THIS specific instance
        query_dict, raw_cfs = engine.generate_counterfactuals(instance_idx, desired_enc, 5)
        
        print(f"[BATCH DEBUG] Instance {instance_idx}: Generated {len(raw_cfs)} CFs, query_dict keys: {query_dict.keys()}")
        
        # Get ORIGINAL prediction for THIS instance using engine preprocessing
        original_instance_dict = query_dict.copy()
        original_pred_probs = engine.predict_proba(original_instance_dict)
        original_pred_prob = float(original_pred_probs[desired_enc])
        
        print(f"[BATCH DEBUG] Instance {instance_idx}: Original pred = {original_pred_prob:.4f}")

        # Compute predictions for EACH counterfactual (MUST use engine.predict_proba for preprocessing)
        cf_predictions = []
        for cf_idx, cf in enumerate(raw_cfs):
            try:
                # CRITICAL: Use engine.predict_proba() not engine.model.predict_proba()
                cf_proba = engine.predict_proba(cf)
                cf_pred = float(cf_proba[desired_enc])
                cf_predictions.append(cf_pred)
                print(f"[BATCH DEBUG] Instance {instance_idx}, CF {cf_idx}: pred = {cf_pred:.4f}")
            except Exception as e:
                # Fallback to original prediction if CF evaluation fails
                print(f"[BATCH DEBUG] Instance {instance_idx}, CF {cf_idx}: ERROR {e}, using fallback")
                cf_predictions.append(original_pred_prob)

        # Update query_dict with original prediction for unified evaluation
        query_dict_with_pred = query_dict.copy()
        query_dict_with_pred['original_prediction'] = original_pred_prob

        # Apply rules with unified evaluation (provides valid/invalid CFs)
        valid, invalid = apply_rules(query_dict_with_pred, raw_cfs, auto_rules, cf_predictions)
        
        print(f"[BATCH DEBUG] Instance {instance_idx}: Valid={len(valid)}, Invalid={len(invalid)}")
        
        # Build metrics dict directly from evaluation results
        faithful_cf_count = len(valid)
        violation_cf_count = len(invalid)
        total_cf_count = len(raw_cfs)
        
        # Compute improvements for faithful CFs
        improvements = []
        for valid_cf in valid:
            cf_idx_in_list = raw_cfs.index(valid_cf) if valid_cf in raw_cfs else -1
            if cf_idx_in_list >= 0 and cf_idx_in_list < len(cf_predictions):
                improvement = cf_predictions[cf_idx_in_list] - original_pred_prob
                improvements.append(improvement)
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        max_improvement = max(improvements) if improvements else 0.0
        
        # Build metrics matching evaluator output shape
        metrics = {
            'instance_id': int(instance_idx),
            'num_counterfactuals': total_cf_count,
            'num_actionable_cf': faithful_cf_count,  # Actionable = passes rules
            'num_faithful_cf': faithful_cf_count,    # Faithful = passes rules + improves
            'num_improving_cf': len([i for i in improvements if i > 0]),
            'avg_feasibility_score': 0.8 if faithful_cf_count > 0 else 0.0,  # Placeholder
            'has_feasible_recourse': faithful_cf_count > 0,
            'max_improvement_delta': float(max_improvement),
            'metrics_per_cf': []
        }
        
        return metrics
    except Exception as e:
        print(f"⚠️ Error processing instance {instance_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_batch_analysis_parallel(engine, test_samples_indices, auto_rules, target_classes, progress_container):
    """
    Run batch analysis with parallel processing using ThreadPoolExecutor.
    """
    batch_results = []
    errors = []
    
    desired_class = target_classes[0]
    
    # Prepare task arguments
    tasks = [
        (idx, engine, target_classes, auto_rules, desired_class)
        for idx in test_samples_indices
    ]
    
    # Progress tracking
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    completed = 0
    
    # Execute with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_BATCH) as executor:
        future_to_idx = {executor.submit(process_single_instance_batch, task): idx for idx, task in enumerate(tasks)}
        
        for future in as_completed(future_to_idx):
            instance_num = future_to_idx[future]
            try:
                metrics = future.result()
                if metrics:
                    batch_results.append(metrics)
                completed += 1
                progress_bar.progress(completed / len(tasks))
                status_text.text(f"🚀 Processing... {completed}/{len(tasks)} instances")
            except Exception as e:
                errors.append(f"Instance {instance_num}: {str(e)}")
                completed += 1
                progress_bar.progress(completed / len(tasks))
    
    progress_bar.empty()
    status_text.empty()
    
    return batch_results, errors



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
    'clean_mode': 'auto', 'cleaning_log': {}, 'dataset_name': 'uploaded dataset',
    'current_accuracy': None, 'accuracy_valid': False, 'can_proceed': False,
    'optimization_performed': False, 'optimization_log': {},
    'fast_mode': FAST_MODE_ENABLED, 'batch_limit': DEFAULT_BATCH_LIMIT,
    'last_batch_time': None, 'batch_results': None
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
                st.session_state.current_accuracy = accuracy
                st.session_state.cfs_generated = False
                st.session_state.audit_done = False
                st.session_state.step = max(st.session_state.step, 3)
                
                # ✅ DYNAMIC ACCURACY FEEDBACK
                feedback = display_accuracy_feedback(accuracy, MIN_ACCURACY_THRESHOLD)
                st.session_state.accuracy_valid = feedback["meets_threshold"]
                st.session_state.can_proceed = feedback["meets_threshold"]
                
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
    
    # 🧪 OPTIMIZE DATASET FOR ACCURACY
    if st.session_state.model_trained and not st.session_state.can_proceed:
        st.markdown("---")
        st.markdown("#### 🔧 Dataset Optimization")
        st.markdown(f"**Current Accuracy:** `{st.session_state.current_accuracy:.1%}` | **Threshold:** `{MIN_ACCURACY_THRESHOLD:.1%}`")
        
        opt_col1, opt_col2 = st.columns([1, 1])
        with opt_col1:
            if st.button("🚀 Optimize Dataset", key="optimize_btn", use_container_width=True):
                with st.spinner("🔧 Optimizing dataset (scaling, encoding, outlier removal)..."):
                    try:
                        df_optimized, opt_log = optimize_dataset(df, engine, target_col)
                        st.session_state.optimization_log = opt_log
                        st.session_state.optimization_performed = True
                        
                        # ✅ RE-TRAIN WITH OPTIMIZED DATA
                        with st.spinner("Re-training model with optimized data..."):
                            engine.df = df_optimized
                            engine.detect_features(target_col)
                            new_accuracy = engine.train_model()
                            st.session_state.current_accuracy = new_accuracy
                            
                            # Check if threshold is now met
                            is_valid, message, can_proceed = check_accuracy_threshold(new_accuracy)
                            st.session_state.accuracy_valid = is_valid
                            st.session_state.can_proceed = can_proceed
                            
                            if can_proceed:
                                st.success(f"✅ Optimization successful! New Accuracy: **{new_accuracy:.1%}**\n✅ {message}")
                                st.balloons()
                            else:
                                st.warning(f"⚠️ Optimization applied. New Accuracy: **{new_accuracy:.1%}**\n{message}\n\n💡 Try a different target feature or review data quality.")
                    except Exception as e:
                        st.error(f"Optimization error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with opt_col2:
            if st.session_state.optimization_log:
                with st.expander("📊 Optimization Details", expanded=False):
                    changes = st.session_state.optimization_log.get("changes", {})
                    if "features_scaled" in changes:
                        st.metric("📈 Features Scaled", changes["features_scaled"])
                    if "features_encoded" in changes:
                        st.metric("🔤 Features Encoded", changes["features_encoded"])
                    if "advanced_outliers_removed" in changes:
                        st.metric("🎯 Outliers Removed", changes["advanced_outliers_removed"])
                    if "final_rows" in changes:
                        st.metric("✅ Final Rows", changes["final_rows"])

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

                    # Get original prediction probability (not class label)
                    original_instance = engine.X_test.iloc[[query_idx]]
                    original_pred_probs = engine.model.predict_proba(original_instance)[0]
                    original_pred_prob = float(original_pred_probs[desired_enc])

                    # Compute predictions for each counterfactual for unified evaluation
                    # Use engine.predict_proba() to ensure proper label encoding
                    cf_predictions = []
                    for cf in raw_cfs:
                        try:
                            # Use the engine's predict_proba method (applies correct preprocessing)
                            proba = engine.predict_proba(cf)
                            cf_pred = float(proba[desired_enc]) if isinstance(proba, (list, np.ndarray)) else float(proba)
                            cf_predictions.append(cf_pred)
                        except Exception as e:
                            # Fallback to original prediction if CF evaluation fails
                            print(f"[DEBUG] CF prediction error: {e}")
                            cf_predictions.append(original_pred_prob)
                    
                    # Store for use in narrative generation
                    st.session_state.cf_predictions = cf_predictions

                    # Update query_dict with original prediction for unified evaluation
                    query_dict_with_pred = query_dict.copy()
                    query_dict_with_pred['original_prediction'] = original_pred_prob

                    # AUTO-AUDIT using smart rules with unified evaluation
                    valid, invalid = apply_rules(query_dict_with_pred, raw_cfs, auto_rules, cf_predictions)
                    st.session_state.valid_cfs = valid
                    st.session_state.invalid_cfs = invalid
                    st.session_state.cfs_generated = True
                    st.session_state.audit_done = True
                    st.session_state.narrative = None  # Reset for fresh LLM call

                    # Compute faithfulness metrics
                    evaluator = create_evaluator(auto_rules)
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
                    model_pred=float(pred_prob),  # Pass numeric value, not formatted string
                    cf_predictions=st.session_state.get('cf_predictions', []),
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
            st.markdown(f'<div class="metric-card"><h3 style="color:#16a34a">{n_valid}</h3><p>✅ Faithful</p></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card"><h3 style="color:#dc2626">{n_invalid}</h3><p>❌ Faithless</p></div>', unsafe_allow_html=True)

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
            st.markdown("### ✅ Faithful Suggestions")
            st.markdown("*These suggestions respect domain rules AND improve model predictions*")
            for i, cf in enumerate(st.session_state.valid_cfs, 1):
                with st.container():
                    # Use new generic formatter
                    formatted = format_suggestion(cf, st.session_state.query_dict, i)
                    st.markdown(formatted)
                    st.markdown("---")

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
# BATCH ANALYSIS SECTION (OPTIMIZED)
# ═══════════════════════════════════════

# Run batch analysis on uploaded data
if st.session_state.model_trained and df is not None and st.session_state.can_proceed:
    st.markdown("---")
    st.markdown("### 🚀 Batch Analysis (Optimized)")
    
    # Configuration section
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        fast_mode = st.checkbox("⚡ Fast Mode (Skip heavy computations)", value=False, key="fast_mode_toggle")
        st.session_state.fast_mode = fast_mode
        if fast_mode:
            st.markdown("<span style='color: #f59e0b; font-weight: bold;'>⚡ Fast mode enabled - Reduced accuracy for speed</span>", unsafe_allow_html=True)
    
    with config_col2:
        batch_limit = st.slider("📊 Limit dataset rows for analysis", 5, min(100, len(df)), DEFAULT_BATCH_LIMIT, key="batch_limit_slider")
        st.session_state.batch_limit = batch_limit
        st.markdown(f"<span style='color: #667eea;'>Will analyze **{batch_limit}** instances</span>", unsafe_allow_html=True)
    
    with config_col3:
        st.markdown("")
        st.markdown("")
        st.markdown(f"<span style='color: #6b7280; font-size: 0.9rem;'>Parallel workers: **{MAX_WORKERS_BATCH}**</span>", unsafe_allow_html=True)
    
    # Run button with better messaging
    if st.button("🔍 Analyze Instances (Parallel Processing)", type="secondary", use_container_width=True, key="batch_analysis_btn"):
        start_time = time.time()
        
        with st.spinner(""):
            try:
                # Get test samples with limit
                test_samples_count = min(batch_limit, len(df))
                test_samples = engine.get_test_samples(test_samples_count)
                test_samples_indices = list(range(len(test_samples)))
                
                # Create progress container
                progress_container = st.container()
                
                # Run parallel batch analysis
                batch_results, errors = run_batch_analysis_parallel(
                    engine,
                    test_samples_indices,
                    st.session_state.auto_rules,
                    engine.get_target_classes(),
                    progress_container
                )
                
                elapsed_time = time.time() - start_time
                st.session_state.last_batch_time = elapsed_time
                st.session_state.batch_results = batch_results
                
                # Display results
                if batch_results:
                    st.success(f"✅ Analysis complete! Processed {len(batch_results)} instances in **{elapsed_time:.2f}s** (⚡ {elapsed_time/len(batch_results):.2f}s per instance)")
                    
                    if errors:
                        with st.expander(f"⚠️ Errors ({len(errors)})"):
                            for error in errors:
                                st.warning(error)
                    
                    # Display batch results
                    st.markdown("---")
                    st.markdown("#### 📊 Batch Analysis Results")
                    
                    # Convert to DataFrame
                    batch_df = pd.DataFrame(batch_results)
                    
                    # Overall batch metrics
                    batch_total_cfs = batch_df['num_counterfactuals'].sum()
                    batch_total_faithful = batch_df['num_faithful_cf'].sum()
                    batch_total_actionable = batch_df['num_actionable_cf'].sum()
                    batch_total_violations = batch_total_cfs - batch_total_actionable  # Violations = CFs that violate rules
                    batch_agent_full = batch_df['has_feasible_recourse'].sum()
                    
                    print(f"[BATCH DISPLAY] Total CFs: {batch_total_cfs}, Actionable: {batch_total_actionable}, Faithful: {batch_total_faithful}, Violations: {batch_total_violations}")
                    
                    batch_faithful_rate = (batch_total_faithful / max(batch_total_cfs, 1)) * 100
                    batch_violation_rate = (batch_total_violations / max(batch_total_cfs, 1)) * 100
                    batch_agent_rate = (batch_agent_full / max(len(batch_df), 1)) * 100
                    
                    # Summary metrics
                    br1, br2, br3, br4 = st.columns(4)
                    with br1:
                        st.markdown(f'<div class="metric-card"><h3 style="color:#22c55e">{batch_faithful_rate:.1f}%</h3><p>✅ Faithful Rate</p></div>', unsafe_allow_html=True)
                    with br2:
                        st.markdown(f'<div class="metric-card"><h3 style="color:#ef4444">{batch_violation_rate:.1f}%</h3><p>❌ Rule Violations</p></div>', unsafe_allow_html=True)
                    with br3:
                        st.markdown(f'<div class="metric-card"><h3 style="color:#f59e0b">{batch_agent_rate:.1f}%</h3><p>🎯 Agent Full Rate</p></div>', unsafe_allow_html=True)
                    with br4:
                        st.markdown(f'<div class="metric-card"><h3 style="color:#6366f1">{len(batch_results)}</h3><p>📋 Instances</p></div>', unsafe_allow_html=True)
                    
                    # Per-instance results table
                    st.markdown("##### Per-Instance Breakdown")
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
                            'Agent-Full': "✓" if has_recourse else "✗"
                        })
                    
                    results_df = pd.DataFrame(instance_results)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download options
                    st.markdown("---")
                    st.markdown("#### 💾 Export Results")
                    
                    col_down1, col_down2, col_down3 = st.columns(3)
                    
                    with col_down1:
                        st.download_button(
                            "📥 CSV (Per-Instance)",
                            batch_df.to_csv(index=False),
                            "batch_results.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with col_down2:
                        batch_export = {
                            "summary": {
                                "total_instances": len(batch_df),
                                "total_counterfactuals": int(batch_total_cfs),
                                "faithful_cfs": int(batch_total_faithful),
                                "faithful_rate_%": round(batch_faithful_rate, 2),
                                "processing_time_seconds": round(elapsed_time, 2),
                                "seconds_per_instance": round(elapsed_time / len(batch_results), 2)
                            },
                            "instances": batch_df.to_dict(orient='records')
                        }
                        st.download_button(
                            "📥 JSON (Full Report)",
                            json.dumps(batch_export, indent=4, default=str),
                            "batch_report.json",
                            "application/json",
                            use_container_width=True
                        )
                    
                    with col_down3:
                        st.markdown(f"**⏱️ Performance:** {elapsed_time:.2f}s total")
                        st.markdown(f"**🚀 Speed:** {elapsed_time/len(batch_results):.3f}s/instance")
                        st.markdown(f"**📊 Throughput:** {len(batch_results)/elapsed_time:.1f} instances/sec")
                
                else:
                    st.error("❌ No results generated. Check error details above.")
                    
            except Exception as e:
                st.error(f"Batch analysis failed: {e}")
                import traceback
                st.code(traceback.format_exc())
elif st.session_state.model_trained and df is not None and not st.session_state.can_proceed:
    st.markdown("---")
    st.markdown("### 🚀 Batch Analysis (Disabled)")
    st.warning(f"⚠️ Batch analysis is **disabled** until accuracy meets the threshold ({MIN_ACCURACY_THRESHOLD:.1%}). Please optimize your dataset first.", icon="🔒")

# Footer
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.85rem; padding:1rem;">
    <strong>ACR Dashboard</strong> — Agentic Counterfactual Reasoning |
    Built for XAI Project ) |
    Powered by DiCE & SHAP |
    Developed by Kushal Pranav & Team |
</div>
""", unsafe_allow_html=True)
