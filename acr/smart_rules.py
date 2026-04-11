"""
Smart Causal Engine - Automatically detects which features are immutable
based on semantic analysis of column names and data patterns.
No manual selection needed.
"""

from .evaluation import evaluate_explanation, evaluate_batch_explanations

# Known immutable feature patterns across domains
IMMUTABLE_PATTERNS = {
    # Demographics (cannot change)
    'age': 'Age is a biological constant — cannot be reversed.',
    'gender': 'Gender is an immutable characteristic.',
    'sex': 'Sex is an immutable characteristic.',
    'race': 'Race is an immutable characteristic.',
    'ethnicity': 'Ethnicity is an immutable characteristic.',
    'nationality': 'Nationality is typically immutable in this context.',
    'native_country': 'Native country cannot be changed.',
    'country': 'Country of origin is immutable.',
    'birth': 'Birth-related features are immutable.',
    'dob': 'Date of birth is immutable.',
    'date_of_birth': 'Date of birth is immutable.',

    # Genetics / Biology
    'pedigree': 'Genetic pedigree is immutable.',
    'diabetespedigreefunction': 'Genetic predisposition cannot be changed.',
    'genetic': 'Genetic features are immutable.',
    'dna': 'DNA-based features are immutable.',
    'hereditary': 'Hereditary features are immutable.',
    'family_history': 'Family history cannot be changed.',

    # Historical / Past events
    'pregnancies': 'Past pregnancies cannot be undone.',
    'num_children': 'Number of existing children is historical.',
    'years_experience': 'Past experience years cannot decrease.',
    'tenure': 'Past tenure cannot decrease.',
}

# Known increase-only patterns
INCREASE_ONLY_PATTERNS = {
    'education': 'Education level can only increase (you cannot un-learn).',
    'education_num': 'Education level can only increase.',
    'qualification': 'Qualifications can only increase.',
    'degree': 'Degree level can only increase.',
    'experience': 'Experience can only increase over time.',
    'skill_level': 'Skill levels typically only increase.',
}

# Known decrease-only patterns (health improvements)
DECREASE_ONLY_PATTERNS = {
    # Generally empty - most health metrics can go either way
    # But could add domain-specific ones
}


def auto_detect_rules(feature_names, df=None):
    """
    Automatically detect causal rules for each feature based on its name.
    SAFE: Skips features not found in dataframe.
    
    Returns:
        rules: dict of {feature_name: {'mutable': bool, 'constraint': str|None, 'reason': str}}
    """
    rules = {}
    
    for feat in feature_names:
        # CRITICAL: Skip features that don't exist in the dataframe
        if df is not None and feat not in df.columns:
            print(f"[DEBUG] Skipping feature '{feat}' (not in dataframe columns)")
            continue
        
        feat_lower = feat.lower().strip().replace(' ', '_')
        
        # Check immutable patterns
        matched = False
        for pattern, reason in IMMUTABLE_PATTERNS.items():
            if pattern in feat_lower:
                rules[feat] = {
                    'mutable': False,
                    'constraint': None,
                    'reason': reason,
                    'auto_detected': True
                }
                matched = True
                break
        
        if matched:
            continue
            
        # Check increase-only patterns
        for pattern, reason in INCREASE_ONLY_PATTERNS.items():
            if pattern in feat_lower:
                rules[feat] = {
                    'mutable': True,
                    'constraint': 'increase_only',
                    'reason': reason,
                    'auto_detected': True
                }
                matched = True
                break
        
        if matched:
            continue
        
        # Check decrease-only patterns
        for pattern, reason in DECREASE_ONLY_PATTERNS.items():
            if pattern in feat_lower:
                rules[feat] = {
                    'mutable': True,
                    'constraint': 'decrease_only',
                    'reason': reason,
                    'auto_detected': True
                }
                matched = True
                break
        
        if matched:
            continue
        
        # Additional heuristic: if a feature has very few unique values
        # and looks like an ID or categorical constant, flag it
        if df is not None:
            col_data = df[feat]
            # If column has only 1 unique value, it's constant
            if col_data.nunique() <= 1:
                rules[feat] = {
                    'mutable': False,
                    'constraint': None,
                    'reason': f"Feature '{feat}' has only {col_data.nunique()} unique value(s) — likely constant.",
                    'auto_detected': True
                }
                continue
        
        # Default: feature is mutable with no constraints
        rules[feat] = {
            'mutable': True,
            'constraint': None,
            'reason': 'No known constraints detected.',
            'auto_detected': True
        }
    
    return rules


def apply_rules(query_dict, raw_cfs, rules, predictions=None):
    """
    Filter counterfactuals using unified evaluation system.

    Now uses the centralized evaluate_explanation() function to ensure
    consistency with batch analysis.

    Args:
        query_dict: Original instance data (must include 'original_prediction')
        raw_cfs: List of counterfactual dicts
        rules: Domain rules dict
        predictions: List of predictions for each counterfactual (optional)

    Returns:
        valid_cfs: list of dicts (faithful suggestions)
        invalid_cfs: list of dicts with 'suggestion' and 'reason'
    """
    valid_cfs = []
    invalid_cfs = []

    # Extract original prediction
    original_pred = query_dict.get('original_prediction', 0.5)
    try:
        original_pred = float(original_pred)
    except (ValueError, TypeError):
        original_pred = 0.5

    # If predictions not provided, assume all have same prediction as original
    if predictions is None:
        predictions = [original_pred] * len(raw_cfs)

    if len(predictions) != len(raw_cfs):
        raise ValueError("Predictions list must match counterfactuals list length")

    for cf, prediction in zip(raw_cfs, predictions):
        # Use unified evaluation (NEW SIGNATURE)
        result = evaluate_explanation(cf, prediction, query_dict, original_pred, rules)

        if result.get("faithful", False):
            valid_cfs.append(cf)
        else:
            # Use the reason from evaluate_explanation
            violation_reason = result.get("reason", "❌ Counterfactual violates faithfulness criteria")
            
            invalid_cfs.append({
                "suggestion": cf,
                "reason": violation_reason
            })

    return valid_cfs, invalid_cfs
