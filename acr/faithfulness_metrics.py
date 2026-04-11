"""
Faithfulness metrics for ACR-PS.
Computes instance-level and dataset-level evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class FaithfulnessMetrics:
    """
    Container for faithfulness evaluation metrics.
    """
    # Instance-level metrics (aggregated to dataset)
    faithful_rate: float  # % of suggestions that obey rules AND improve prediction
    rule_violation_rate: float  # % of suggestions that change immutable/invalid-direction features
    agent_full_rate: float  # % of instances with >=1 feasible, actionable, faithful recourse
    
    # Counterfactual quality
    avg_valid_counterfactuals: float  # Average # of valid CFs per instance
    avg_invalid_counterfactuals: float  # Average # of invalid CFs per instance
    
    # Improvement
    prediction_improvement_rate: float  # % of suggestions that improve model prediction
    
    # Additional context
    num_instances: int
    num_counterfactuals_total: int
    num_actionable: int
    num_faithful: int


class FaithfulnessEvaluator:
    """
    Compute faithfulness metrics at instance and dataset levels.
    """
    
    def __init__(self, rules: Optional[Dict] = None):
        """
        Args:
            rules: Dict of rules (feature-level constraints)
        """
        self.rules = rules or {}
    
    def compute_instance_faithfulness(
        self,
        instance_id: Any,
        original_features: pd.Series,
        counterfactuals: List[pd.Series],
        original_prediction: float,
        model,
        desired_class: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute faithfulness metrics for a single instance.
        
        Args:
            instance_id: Identifier for the instance
            original_features: Original feature values (Series)
            counterfactuals: List of counterfactual feature Series
            original_prediction: Model's prediction on original
            model: Trained model (sklearn-like)
            desired_class: Target class for improvement (default 1)
        
        Returns:
            instance_metrics: Dict with keys:
                - instance_id
                - num_counterfactuals
                - num_actionable_cf
                - num_faithful_cf
                - num_improving_cf
                - avg_feasibility_score
                - has_feasible_recourse (bool)
                - max_improvement_delta
                - metrics_per_cf: List[Dict] with per-CF details
        """
        if not counterfactuals:
            return {
                'instance_id': instance_id,
                'num_counterfactuals': 0,
                'num_actionable_cf': 0,
                'num_faithful_cf': 0,
                'num_improving_cf': 0,
                'avg_feasibility_score': 0.0,
                'has_feasible_recourse': False,
                'max_improvement_delta': 0.0,
                'metrics_per_cf': [],
            }
        
        metrics_per_cf = []
        actionable_count = 0
        faithful_count = 0
        improving_count = 0
        feasibility_scores = []
        improvement_deltas = []
        
        for cf_idx, cf in enumerate(counterfactuals):
            # Evaluate actionability
            actionability = self._evaluate_actionability(original_features, cf)
            is_actionable = actionability['is_actionable']
            feasibility_score = actionability['feasibility_score']
            feasibility_scores.append(feasibility_score)
            
            if is_actionable:
                actionable_count += 1
            
            # Evaluate prediction improvement
            try:
                cf_input = cf.values.reshape(1, -1) if len(cf.shape) == 1 else cf
                cf_pred = model.predict_proba(cf_input)[0, desired_class]
            except Exception:
                cf_pred = original_prediction
            
            # Ensure both are numeric floats
            original_pred_numeric = float(original_prediction) if isinstance(original_prediction, str) else original_prediction
            cf_pred_numeric = float(cf_pred) if isinstance(cf_pred, str) else cf_pred
            
            improvement_delta = cf_pred_numeric - original_pred_numeric
            improvement_deltas.append(improvement_delta)
            
            improved = float(improvement_delta) > 0
            if improved:
                improving_count += 1
            
            # Faithful = actionable AND improving
            is_faithful = is_actionable and improved
            if is_faithful:
                faithful_count += 1
            
            metrics_per_cf.append({
                'cf_index': cf_idx,
                'is_actionable': is_actionable,
                'is_faithful': is_faithful,
                'is_improving': improved,
                'feasibility_score': feasibility_score,
                'improvement_delta': improvement_delta,
                'cf_prediction': cf_pred,
                'violations': {
                    'immutables': actionability['violated_immutables'],
                    'directions': actionability['invalid_directions'],
                }
            })
        
        avg_feasibility = np.mean(feasibility_scores) if feasibility_scores else 0.0
        max_improvement = max(improvement_deltas) if improvement_deltas else 0.0
        has_feasible = faithful_count > 0
        
        return {
            'instance_id': instance_id,
            'num_counterfactuals': len(counterfactuals),
            'num_actionable_cf': actionable_count,
            'num_faithful_cf': faithful_count,
            'num_improving_cf': improving_count,
            'avg_feasibility_score': float(avg_feasibility),
            'has_feasible_recourse': bool(has_feasible),
            'max_improvement_delta': float(max_improvement),
            'metrics_per_cf': metrics_per_cf,
        }
    
    def compute_dataset_faithfulness(
        self,
        instance_records: List[Dict]
    ) -> Tuple[pd.DataFrame, FaithfulnessMetrics]:
        """
        Aggregate instance-level metrics to dataset level.
        
        Args:
            instance_records: List of dicts from compute_instance_faithfulness
        
        Returns:
            instance_df: DataFrame of per-instance metrics
            summary_metrics: FaithfulnessMetrics object
        """
        instance_dicts = []
        
        for rec in instance_records:
            num_cf = rec['num_counterfactuals']
            num_actionable = rec['num_actionable_cf']
            num_faithful = rec['num_faithful_cf']
            num_improving = rec['num_improving_cf']
            
            instance_dicts.append({
                'instance_id': rec['instance_id'],
                'num_counterfactuals': num_cf,
                'num_actionable': num_actionable,
                'num_faithful': num_faithful,
                'num_improving': num_improving,
                'has_feasible_recourse': rec['has_feasible_recourse'],
                'avg_feasibility_score': rec['avg_feasibility_score'],
                'max_improvement_delta': rec['max_improvement_delta'],
            })
        
        instance_df = pd.DataFrame(instance_dicts)
        
        # Compute dataset-level aggregates
        num_instances = len(instance_df)
        num_cf_total = instance_df['num_counterfactuals'].sum()
        num_actionable_total = instance_df['num_actionable'].sum()
        num_faithful_total = instance_df['num_faithful'].sum()
        num_improving_total = instance_df['num_improving'].sum()
        
        faithful_rate = (num_faithful_total / max(num_cf_total, 1)) * 100.0
        rule_violation_rate = ((num_actionable_total - num_faithful_total) / max(num_cf_total, 1)) * 100.0
        agent_full_rate = (instance_df['has_feasible_recourse'].sum() / max(num_instances, 1)) * 100.0
        prediction_improvement_rate = (num_improving_total / max(num_cf_total, 1)) * 100.0
        
        # Average valid/invalid CFs per instance
        valid_cf_mask = (instance_df['num_actionable'] > 0)
        avg_valid_per_instance = instance_df[valid_cf_mask]['num_actionable'].mean() if valid_cf_mask.any() else 0.0
        invalid_cf_per_instance = instance_df['num_counterfactuals'] - instance_df['num_actionable']
        avg_invalid_per_instance = invalid_cf_per_instance.mean()
        
        summary = FaithfulnessMetrics(
            faithful_rate=faithful_rate,
            rule_violation_rate=rule_violation_rate,
            agent_full_rate=agent_full_rate,
            avg_valid_counterfactuals=avg_valid_per_instance,
            avg_invalid_counterfactuals=avg_invalid_per_instance,
            prediction_improvement_rate=prediction_improvement_rate,
            num_instances=num_instances,
            num_counterfactuals_total=num_cf_total,
            num_actionable=num_actionable_total,
            num_faithful=num_faithful_total,
        )
        
        return instance_df, summary
    
    def _evaluate_actionability(
        self,
        original: pd.Series,
        counterfactual: pd.Series
    ) -> Dict[str, Any]:
        """
        Check if counterfactual respects rules (immutability, direction constraints).
        """
        actionability = {
            'is_actionable': True,
            'violated_immutables': [],
            'invalid_directions': [],
            'feasibility_score': 1.0,
        }
        
        for feature in original.index:
            if feature not in self.rules:
                continue
            
            rule = self.rules[feature]
            orig_val = original[feature]
            cf_val = counterfactual[feature]
            
            # Check immutability
            if not rule.get('is_mutable', True):
                if orig_val != cf_val:
                    actionability['violated_immutables'].append(feature)
                    actionability['is_actionable'] = False
            
            # Check direction for numeric features
            if rule.get('type') == 'numeric' and rule.get('direction') in ['increase', 'decrease']:
                try:
                    change = float(cf_val) - float(orig_val)
                    if rule['direction'] == 'increase' and change < 0:
                        actionability['invalid_directions'].append(feature)
                        actionability['is_actionable'] = False
                    elif rule['direction'] == 'decrease' and change > 0:
                        actionability['invalid_directions'].append(feature)
                        actionability['is_actionable'] = False
                except (ValueError, TypeError):
                    pass
        
        # Compute feasibility score based on actionability
        if not actionability['is_actionable']:
            actionability['feasibility_score'] = 0.0
        
        return actionability
    
    def to_summary_dict(self, metrics: FaithfulnessMetrics) -> Dict[str, float]:
        """
        Convert FaithfulnessMetrics to plain dict for easy export.
        """
        return asdict(metrics)


def create_evaluator(rules: Optional[Dict] = None) -> FaithfulnessEvaluator:
    """
    Factory function.
    """
    return FaithfulnessEvaluator(rules=rules)