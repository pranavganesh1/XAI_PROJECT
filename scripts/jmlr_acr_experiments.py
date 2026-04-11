"""
JMLR Reproducibility Script: ACR-PS Experiments on Multiple Datasets
Runs ACR-PS pipeline on: Adult, Diabetes, German Credit, Bank Marketing, COMPAS
Saves results to results/jmlr/
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import traceback
from typing import Dict

warnings.filterwarnings('ignore')

# Import from ACR package (or fallback to local versions)
try:
    from acr.data_loader import load_and_prepare
    from acr.engine import ACREngineExtended, create_engine
    from acr.faithfulness_metrics import create_evaluator
except ImportError:
    print("Warning: Could not import from acr package. Using local fallbacks.")
    # Local imports (for development)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from acr.data_loader import load_and_prepare
        from acr.engine import ACREngine
        from acr.faithfulness_metrics import create_evaluator
        # Define missing functions/classes
        ACREngineExtended = ACREngine
        def create_engine():
            return ACREngine()
    except ImportError as e:
        print(f"Error: Could not import ACR modules. {e}")
        sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JMLRExperimentRunner:
    """
    Main runner for JMLR reproducibility experiments.
    """
    
    DATASETS = ['adult', 'diabetes', 'german_credit', 'bank_marketing', 'compas']
    OUTPUT_DIR = Path('results/jmlr')
    
    def __init__(self, output_dir: Path = None, sample_size: int = 50):
        """
        Args:
            output_dir: Path to save results (default: results/jmlr/)
            sample_size: Number of instances to process per dataset
        """
        self.output_dir = output_dir or self.OUTPUT_DIR
        self.sample_size = sample_size
        self.results = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def run_all_experiments(self) -> Dict:
        """
        Run experiments on all datasets.
        
        Returns:
            results: Dict mapping dataset_name -> experiment_results
        """
        all_summaries = []
        
        for dataset_name in self.DATASETS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment on {dataset_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                dataset_results = self.run_single_experiment(dataset_name)
                self.results[dataset_name] = dataset_results
                
                if dataset_results['summary'] is not None:
                    summary_row = {
                        'dataset': dataset_name,
                        **dataset_results['summary']
                    }
                    all_summaries.append(summary_row)
                    
            except Exception as e:
                logger.error(f"Error in {dataset_name}: {e}")
                traceback.print_exc()
                self.results[dataset_name] = {
                    'error': str(e),
                    'instance_df': None,
                    'summary': None,
                }
        
        # Save all-datasets summary
        if all_summaries:
            all_summary_df = pd.DataFrame(all_summaries)
            summary_path = self.output_dir / 'all_datasets_summary.csv'
            all_summary_df.to_csv(summary_path, index=False)
            logger.info(f"\nSaved all-datasets summary to {summary_path}")
            
            # Print summary table
            self._print_summary_table(all_summary_df)
        
        return self.results
    
    def run_single_experiment(self, dataset_name: str) -> Dict:
        """
        Run ACR-PS experiment on a single dataset.
        
        Args:
            dataset_name: str, one of DATASETS
        
        Returns:
            result: Dict with keys:
                - instance_df: DataFrame of per-instance metrics
                - summary: Dict of aggregated metrics
                - error: str (if failed)
        """
        result = {'instance_df': None, 'summary': None, 'error': None}
        
        # Load dataset
        logger.info(f"Loading {dataset_name}...")
        try:
            X, y, target_col = load_and_prepare(dataset_name)
            logger.info(f"  Shape: X={X.shape}, y={y.shape}")
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            result['error'] = f"Data loading failed: {e}"
            return result
        
        # Encode categorical features
        X_encoded = self._encode_categoricals(X)
        
        # Train baseline model
        logger.info("Training baseline RandomForest model...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.3, random_state=42, stratify=y
            )
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            logger.info(f"  Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            result['error'] = f"Model training failed: {e}"
            return result
        
        # Setup ACR engine
        logger.info("Setting up ACR engine...")
        try:
            engine = create_engine()
            rules = engine.setup_rules(X_encoded, y)
            logger.info(f"  Detected {len(rules)} feature rules")
        except Exception as e:
            logger.warning(f"Rule setup failed (continuing anyway): {e}")
            rules = {}
        
        # Sample instances
        if len(X_test) > self.sample_size:
            sample_indices = np.random.choice(len(X_test), self.sample_size, replace=False)
            X_sample = X_test.iloc[sample_indices].reset_index(drop=True)
            y_sample = y_test.iloc[sample_indices].reset_index(drop=True)
        else:
            X_sample = X_test.reset_index(drop=True)
            y_sample = y_test.reset_index(drop=True)
        
        logger.info(f"Sampled {len(X_sample)} instances for evaluation")
        
        # Compute faithfulness metrics
        logger.info("Computing faithfulness metrics...")
        evaluator = create_evaluator(rules=rules)
        instance_metrics = []
        
        for idx in range(len(X_sample)):
            instance = X_sample.iloc[idx]
            original_pred = model.predict_proba([instance.values])[0, 1]
            
            # Generate synthetic counterfactuals (for demo)
            # In practice, you would use ACR's counterfactual generator
            counterfactuals = self._generate_demo_counterfactuals(instance, num_cf=5)
            
            metrics = evaluator.compute_instance_faithfulness(
                instance_id=idx,
                original_features=instance,
                counterfactuals=counterfactuals,
                original_prediction=original_pred,
                model=model,
                desired_class=1,
            )
            instance_metrics.append(metrics)
        
        # Aggregate metrics
        instance_df, summary = evaluator.compute_dataset_faithfulness(instance_metrics)
        
        # Save results
        instance_path = self.output_dir / f'{dataset_name}_instance_metrics.csv'
        instance_df.to_csv(instance_path, index=False)
        logger.info(f"Saved instance metrics to {instance_path}")
        
        # Save summary
        summary_dict = evaluator.to_summary_dict(summary)
        summary_path = self.output_dir / f'{dataset_name}_summary.csv'
        pd.DataFrame([summary_dict]).to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        # Print summary
        logger.info("\nDataset-level metrics:")
        for key, val in summary_dict.items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.2f}")
            else:
                logger.info(f"  {key}: {val}")
        
        result['instance_df'] = instance_df
        result['summary'] = summary_dict
        return result
    
    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Simple categorical encoding.
        """
        X_encoded = X.copy()
        
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        return X_encoded
    
    def _generate_demo_counterfactuals(
        self,
        instance: pd.Series,
        num_cf: int = 5,
        noise_std: float = 0.1
    ) -> list:
        """
        Generate simple synthetic counterfactuals for demo.
        In real usage, use ACR's counterfactual generator.
        
        Args:
            instance: Original feature values
            num_cf: Number of counterfactuals to generate
            noise_std: Std dev of Gaussian noise
        
        Returns:
            counterfactuals: List of pd.Series
        """
        counterfactuals = []
        
        for _ in range(num_cf):
            # Small perturbations
            cf = instance.copy()
            numeric_cols = cf[cf.apply(lambda x: isinstance(x, (int, float)))].index
            
            for col in numeric_cols:
                noise = np.random.normal(0, noise_std * abs(cf[col]) + 0.01)
                cf[col] = cf[col] + noise
            
            counterfactuals.append(cf)
        
        return counterfactuals
    
    def _print_summary_table(self, summary_df: pd.DataFrame):
        """
        Print a formatted summary table to console.
        """
        print("\n" + "="*100)
        print("JMLR EXPERIMENTS SUMMARY")
        print("="*100)
        
        # Select key metrics for display
        key_cols = [
            'dataset',
            'faithful_rate',
            'agent_full_rate',
            'prediction_improvement_rate',
            'num_instances',
        ]
        
        display_df = summary_df[[col for col in key_cols if col in summary_df.columns]]
        
        # Format floats
        for col in display_df.columns:
            if display_df[col].dtype == 'float64':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        
        print(display_df.to_string(index=False))
        print("="*100 + "\n")


def main(datasets: list = None, sample_size: int = 50, output_dir: str = None):
    """
    Main entry point.
    
    Args:
        datasets: List of dataset names (default: all)
        sample_size: Number of instances to process per dataset
        output_dir: Output directory path
    """
    runner = JMLRExperimentRunner(
        output_dir=Path(output_dir) if output_dir else None,
        sample_size=sample_size
    )
    
    if datasets:
        runner.DATASETS = datasets
    
    results = runner.run_all_experiments()
    
    logger.info("\nExperiment run complete!")
    logger.info(f"Results saved to {runner.output_dir}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='JMLR ACR-PS Experiments')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=None,
                       help='Datasets to run (default: all)')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='Number of instances per dataset')
    parser.add_argument('--output-dir', type=str, default='results/jmlr',
                       help='Output directory')
    
    args = parser.parse_args()
    
    main(
        datasets=args.datasets,
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )