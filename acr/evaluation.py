"""
Unified Evaluation System for ACR Dashboard
Central evaluation logic for counterfactual explanations.
Ensures consistency across batch analysis and audit results.
"""

import logging
from typing import Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_explanation(counterfactual: Dict[str, Any],
                        counterfactual_pred: float,
                        original_data: Dict[str, Any],
                        original_pred: float,
                        rules: Optional[Dict] = None,
                        threshold: float = 0.1) -> Dict[str, Any]:
    """
    UNIFIED evaluation function - used EVERYWHERE for consistency.
    
    Evaluates whether a counterfactual is "faithful" or a "violation".
    A counterfactual is faithful if it:
    1. Respects all domain rules (immutability, directional constraints)
    2. Improves the model's prediction toward the desired outcome
    
    Args:
        counterfactual: Suggested feature changes (dict)
        counterfactual_pred: Model's prediction for CF
        original_data: Original instance features (dict)
        original_pred: Model's prediction for original
        rules: Domain rules dict (optional)
        threshold: Minimum improvement required (default 0.1 = 10%)
    
    Returns:
        Dict with keys: faithful (bool), reason (str), improvement (float)
    """
    logger.debug("=== EVALUATION START ===")
    logger.debug(f"Original pred: {original_pred:.4f}, CF pred: {counterfactual_pred:.4f}")

    # 1. Validate inputs
    if not counterfactual or not isinstance(counterfactual, dict):
        logger.warning("Empty or invalid counterfactual")
        return {"faithful": False, "reason": "Empty counterfactual", "improvement": 0.0}

    # 1b. Check if counterfactual has actual feature changes
    has_changes = False
    for feature, cf_val in counterfactual.items():
        orig_val = original_data.get(feature)
        if orig_val is not None:
            try:
                if float(orig_val) != float(cf_val):
                    has_changes = True
                    break
            except (ValueError, TypeError):
                if str(orig_val) != str(cf_val):
                    has_changes = True
                    break
    
    if not has_changes:
        logger.info("❌ No meaningful feature changes: All features remain the same")
        return {"faithful": False, "reason": "No feature changes", "improvement": 0.0}

    # 2. Check prediction improvement FIRST
    try:
        original_pred_num = float(original_pred)
        cf_pred_num = float(counterfactual_pred)
        improvement = cf_pred_num - original_pred_num
        
        logger.debug(f"Improvement: {improvement:.4f}")

        if improvement <= threshold:
            logger.info(f"❌ No meaningful improvement: {improvement:.4f} <= {threshold}")
            return {"faithful": False, "reason": "No prediction improvement", "improvement": improvement}

    except (ValueError, TypeError) as e:
        logger.error(f"Prediction conversion error: {e}")
        return {"faithful": False, "reason": "Invalid prediction", "improvement": 0.0}

    # 3. Check rule compliance
    rule_violation = check_rule_compliance(counterfactual, original_data, rules)
    if rule_violation:
        logger.info(f"❌ Rule violation: {rule_violation}")
        return {"faithful": False, "reason": rule_violation, "improvement": improvement}

    logger.info(f"✅ FAITHFUL - Improvement: {improvement:.4f}")
    return {"faithful": True, "reason": "Meets all criteria", "improvement": improvement}


def check_rule_compliance(counterfactual: Dict[str, Any],
                         original_data: Dict[str, Any],
                         rules: Optional[Dict] = None) -> Optional[str]:
    """
    Check if counterfactual respects domain rules.
    
    Returns:
        None if compliant, str error message if violation
    """
    if rules is None:
        rules = get_default_rules()

    for feature, rule in rules.items():
        if feature not in counterfactual or feature not in original_data:
            continue

        original_val = original_data[feature]
        cf_val = counterfactual[feature]

        # Skip if unchanged
        if _values_equal(original_val, cf_val):
            continue

        is_mutable = rule.get('mutable', True)
        constraint = rule.get('constraint')

        # Check immutability
        if not is_mutable:
            return f"Cannot change immutable feature '{feature}' ({original_val}→{cf_val})"

        # Check directional constraints
        if constraint == 'increase_only':
            try:
                if float(cf_val) < float(original_val):
                    return f"Feature '{feature}' can only increase ({original_val}→{cf_val})"
            except (ValueError, TypeError):
                pass

        elif constraint == 'decrease_only':
            try:
                if float(cf_val) > float(original_val):
                    return f"Feature '{feature}' can only decrease ({original_val}→{cf_val})"
            except (ValueError, TypeError):
                pass

    return None  # No violations


def _values_equal(val1: Any, val2: Any) -> bool:
    """Check if two values are equal, handling different types."""
    try:
        return float(val1) == float(val2)
    except (ValueError, TypeError):
        return str(val1).strip() == str(val2).strip()


def get_default_rules() -> Dict[str, Dict]:
    """Get default domain rules"""
    return {
        'age': {'mutable': False},
        'sex': {'mutable': False},
        'gender': {'mutable': False},
        'race': {'mutable': False},
        'education': {'type': 'numeric', 'constraint': 'increase_only'},
        'education_num': {'type': 'numeric', 'constraint': 'increase_only'},
        'income': {'type': 'numeric', 'constraint': 'increase_only'},
        'salary': {'type': 'numeric', 'constraint': 'increase_only'},
        'debt': {'type': 'numeric', 'constraint': 'decrease_only'},
    }


def evaluate_batch_explanations(counterfactuals: list,
                               predictions: list,
                               original_data: Dict[str, Any],
                               original_pred: float,
                               rules: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Evaluate multiple counterfactuals - USED IN BATCH PROCESSING.
    Returns consistent metrics.
    """
    if not counterfactuals or not predictions:
        return {
            "total": 0,
            "faithful_count": 0,
            "violation_count": 0,
            "avg_improvement": 0.0,
            "faithful_rate": 0.0,
        }

    results = []
    for cf, cf_pred in zip(counterfactuals, predictions):
        result = evaluate_explanation(cf, cf_pred, original_data, original_pred, rules)
        results.append(result)

    faithful_count = sum(1 for r in results if r.get("faithful", False))
    improvements = [r.get("improvement", 0.0) for r in results]

    return {
        "total": len(results),
        "faithful_count": faithful_count,
        "violation_count": len(results) - faithful_count,
        "avg_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
        "faithful_rate": faithful_count / len(results) if results else 0.0,
        "details": results,
    }
