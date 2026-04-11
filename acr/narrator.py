"""
Dynamic Explanation Generator - Creates instance-specific narratives
based on actual predictions and counterfactuals (not hardcoded text).
"""

import json
import logging

logger = logging.getLogger(__name__)


def generate_explanation(instance_id, query_dict, valid_cfs, invalid_cfs, 
                        feature_names, model_pred=None, cf_predictions=None, 
                        dataset_name="dataset"):
    """
    Generate DYNAMIC, instance-specific explanation.
    
    Args:
        instance_id: Row identifier
        query_dict: Original feature values(dict)
        valid_cfs: Valid counterfactuals (list of dicts)
        invalid_cfs: Invalid suggestions with reasons (list)
        feature_names: List of feature names
        model_pred: Original model prediction probability
        cf_predictions: Predictions for each counterfactual (list)
        dataset_name: Dataset name for context
    
    Returns:
        explanation: Dynamic narrative tailored to THIS instance
    """
    logger.info(f"Generating explanation for Instance #{instance_id}")
    
    # Build narrative
    narrative = f"**Instance #{instance_id}**\n"
    
    if model_pred is not None:
        try:
            # Ensure model_pred is numeric
            pred_numeric = float(model_pred) if not isinstance(model_pred, float) else model_pred
            narrative += f"• Current Probability: {pred_numeric:.2%}\n"
        except (ValueError, TypeError):
            narrative += "• Current Prediction: Available\n"
    
    # If no valid CFs
    if not valid_cfs:
        if invalid_cfs:
            reasons = [item.get('reason', 'Unknown constraint') if isinstance(item, dict) 
                      else str(item) for item in invalid_cfs[:2]]
            narrative += f"• ❌ No feasible actions: {reasons[0]}\n"
        else:
            narrative += "• ❌ No suggestions available\n"
        return narrative
    
    # Show number of options
    narrative += f"• ✅ {len(valid_cfs)} feasible option(s) available\n"
    
    # Show top 3 options with actual predictions
    narrative += "\n**Recommended Actions:**\n"
    for idx, cf in enumerate(valid_cfs[:3], 1):
        changes = _get_feature_changes(query_dict, cf)
        
        if changes:
            change_str = ", ".join([f"{k}: {v}" for k, v in list(changes.items())[:2]])
            narrative += f"{idx}. {change_str}"
            
            # Show expected improvement if predictions available
            if cf_predictions and idx <= len(cf_predictions):
                cf_pred = cf_predictions[idx - 1]
                if model_pred and isinstance(cf_pred, (int, float)) and isinstance(model_pred, (int, float)):
                    improvement = cf_pred - model_pred
                    narrative += f" → Expected: {cf_pred:.2%} ({improvement:+.2%})"
            narrative += "\n"
    
    return narrative


def _get_feature_changes(original, modified):
    """Extract meaningful feature changes."""
    changes = {}
    for feat in original.keys():
        if feat not in modified:
            continue
        
        orig_val = original[feat]
        mod_val = modified[feat]
        
        if orig_val != mod_val:
            try:
                # Try numeric formatting
                orig_num = float(orig_val)
                mod_num = float(mod_val)
                pct = ((mod_num - orig_num) / orig_num * 100) if orig_num != 0 else 0
                changes[feat] = f"{orig_num:.1f}→{mod_num:.1f} ({pct:+.0f}%)"
            except (ValueError, TypeError):
                # Categorical change
                changes[feat] = f"{orig_val}→{mod_val}"
    
    return changes


def evaluate_explanation(explanation_text, valid_cfs, invalid_cfs):
    """
    Evaluate if explanation is meaningful.
    
    Returns:
        is_meaningful: bool
        quality_score: float (0-1)
    """
    if not explanation_text or len(explanation_text.strip()) < 10:
        return False, 0.0
    
    # Check specificity
    has_specificity = any(keyword in explanation_text.lower() 
                         for keyword in ['instance', 'option', '→', 'action', 'probability', 'feasible'])
    
    # Check if reflects actual CFs
    if valid_cfs:
        reflects_data = 'feasible' in explanation_text.lower() or 'action' in explanation_text.lower()
    else:
        reflects_data = 'no' in explanation_text.lower() or '❌' in explanation_text
    
    is_meaningful = has_specificity and reflects_data
    quality = 0.9 if is_meaningful else 0.3
    
    return is_meaningful, quality


def get_narrative(query_dict, valid_cfs, invalid_cfs, feature_names, 
                 model_pred=None, cf_predictions=None):
    """
    High-level narrative explaining audit results.
    
    Args:
        query_dict: Original feature values
        valid_cfs: Valid counterfactuals (list of dicts)
        invalid_cfs: Invalid suggestions (list)
        feature_names: List of feature names
        model_pred: Original model prediction
        cf_predictions: Predictions for each counterfactual
        
    Returns:
        Narrative string suitable for end-user display
    """
    
    parts = []
    
    # Opening
    if not valid_cfs and not invalid_cfs:
        parts.append("Insufficient data to generate recommendations.")
    elif not valid_cfs and invalid_cfs:
        parts.append(f"All {len(invalid_cfs)} suggestions required changing immutable features or constraints. No actionable paths available.")
    else:
        parts.append(f"Found {len(valid_cfs)} feasible action(s):")
        
        # List options
        for idx, cf in enumerate(valid_cfs[:3], 1):
            changes = _get_feature_changes(query_dict, cf)
            if changes:
                change_list = ", ".join([f"{k}: {v}" for k, v in list(changes.items())[:2]])
                
                # Add improvement if available
                if cf_predictions and idx <= len(cf_predictions):
                    cf_pred = cf_predictions[idx - 1]
                    if model_pred and isinstance(cf_pred, (int, float)):
                        improvement = cf_pred - model_pred
                        parts.append(f"  Option {idx}: {change_list} → +{improvement:+.1%}")
                    else:
                        parts.append(f"  Option {idx}: {change_list}")
                else:
                    parts.append(f"  Option {idx}: {change_list}")
    
    return " ".join(parts)
