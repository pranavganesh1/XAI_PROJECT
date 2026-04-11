"""
ACR Engine - The core backend for the Agentic Counterfactual Reasoning system.
Handles model training, counterfactual generation, and causal auditing.
Works with ANY dataset from ANY domain.
"""

import pandas as pd
import numpy as np
import dice_ml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import json
import warnings
warnings.filterwarnings('ignore')


class ACREngine:
    """
    Agentic Counterfactual Reasoning Engine.
    Accepts any dataset, trains a model, generates counterfactuals,
    and audits them against user-defined causal rules.
    """

    def __init__(self):
        self.df = None
        self.target = None
        self.feature_names = []
        self.categorical_features = []
        self.continuous_features = []
        self.label_encoders = {}
        self.model = None
        self.dice_data = None
        self.dice_model = None
        self.dice_exp = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.accuracy = 0.0

    # ---- Step 1: Load Data ----
    def load_data(self, uploaded_file):
        """Load data from uploaded file (CSV, Excel, JSON)."""
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            self.df = pd.read_csv(uploaded_file)
        elif name.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(uploaded_file)
        elif name.endswith('.json'):
            self.df = pd.read_json(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {name}")

        # Clean column names
        self.df.columns = [c.strip().replace(' ', '_') for c in self.df.columns]
        # Drop rows with all NaN
        self.df.dropna(how='all', inplace=True)
        return self.df

    def detect_features(self, target_col):
        """Auto-detect categorical vs continuous features."""
        self.target = target_col
        features = [c for c in self.df.columns if c != target_col]
        self.feature_names = features

        self.categorical_features = []
        self.continuous_features = []

        for col in features:
            if self.df[col].dtype == 'object' or self.df[col].nunique() < 10:
                self.categorical_features.append(col)
            else:
                self.continuous_features.append(col)

        return self.categorical_features, self.continuous_features

    # ---- Step 2: Train Model ----
    def train_model(self):
        """Train a RandomForest classifier on the loaded data."""
        df_encoded = self.df.copy()

        # Encode categorical features - STORE for inference
        self.label_encoders = {}
        for col in self.categorical_features:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            self.label_encoders[col] = le

        # Encode target if it's categorical
        if df_encoded[self.target].dtype == 'object':
            le_target = LabelEncoder()
            df_encoded[self.target] = le_target.fit_transform(df_encoded[self.target].astype(str))
            self.label_encoders[self.target] = le_target

        # Handle NaN - STORE statistics for inference
        for col in self.continuous_features:
            if df_encoded[col].isnull().any():
                median_val = df_encoded[col].median()
                df_encoded[col].fillna(median_val, inplace=True)
        for col in self.categorical_features:
            if df_encoded[col].isnull().any():
                mode_val = df_encoded[col].mode()[0] if len(df_encoded[col].mode()) > 0 else 0
                df_encoded[col].fillna(mode_val, inplace=True)

        X = df_encoded[self.feature_names]
        y = df_encoded[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )

        # CRITICAL: Use same preprocessing pipeline for BOTH training and inference
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=15,
                min_samples_split=3,
                n_jobs=-1
            ))
        ])
        self.model.fit(self.X_train, self.y_train)
        self.accuracy = self.model.score(self.X_test, self.y_test)
        
        print(f"✓ Model trained - Accuracy: {self.accuracy:.4f}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Train samples: {len(self.X_train)}")

        return self.accuracy

    # ---- NEW: Proper Prediction with Preprocessing ----
    def predict_proba(self, input_dict):
        """
        Predict using SAME preprocessing pipeline as training.
        This is CRITICAL to ensure predictions change with inputs.
        """
        # Convert dict to DataFrame
        input_df = pd.DataFrame([input_dict])

        # CRITICAL: Encode categorical features SAME way as training
        for col in self.categorical_features:
            if col in self.label_encoders and col in input_df.columns:
                try:
                    input_df[col] = self.label_encoders[col].transform(input_df[col].astype(str))
                except ValueError:
                    # Unknown category - use default
                    input_df[col] = 0

        # CRITICAL: Remove non-feature columns
        input_df = input_df[self.feature_names]

        # CRITICAL: Use predict_proba for sensitivity
        try:
            proba = self.model.predict_proba(input_df)[0]
            return proba
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            n_classes = len(np.unique(self.y_train))
            return np.ones(n_classes) / n_classes

    def predict_proba_class_0(self, input_dict):
        """Get probability for class 0 (negative outcome)"""
        proba = self.predict_proba(input_dict)
        return proba[0] if len(proba) > 0 else 0.5

    def predict_proba_class_1(self, input_dict):
        """Get probability for class 1 (positive outcome)"""
        proba = self.predict_proba(input_dict)
        return proba[1] if len(proba) > 1 else 0.5
    def generate_counterfactuals(self, query_index, desired_class, num_cfs=5):
        """Generate counterfactual explanations using DiCE."""
        df_processed = self.df.copy()

        # Encode data for DiCE
        for col in self.categorical_features:
            if col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        if self.target in self.label_encoders:
            df_processed[self.target] = self.label_encoders[self.target].transform(
                df_processed[self.target].astype(str)
            )

        df_processed.fillna(df_processed.median(numeric_only=True), inplace=True)

        # Setup DiCE
        self.dice_data = dice_ml.Data(
            dataframe=df_processed,
            continuous_features=self.feature_names,  # treat all as continuous for encoded data
            outcome_name=self.target
        )
        self.dice_model = dice_ml.Model(model=self.model, backend="sklearn")
        self.dice_exp = dice_ml.Dice(self.dice_data, self.dice_model, method="random")

        # Pick the query instance
        query_instance = self.X_test.iloc[[query_index]]

        # Generate
        dice_result = self.dice_exp.generate_counterfactuals(
            query_instance, total_CFs=num_cfs, desired_class=int(desired_class)
        )

        # Extract results
        raw_cfs = []
        if dice_result.cf_examples_list and dice_result.cf_examples_list[0].final_cfs_df is not None:
            cf_df = dice_result.cf_examples_list[0].final_cfs_df.copy()
            
            # Decode back to original labels
            for col in self.categorical_features:
                if col in self.label_encoders and col in cf_df.columns:
                    cf_df[col] = cf_df[col].round().astype(int)
                    cf_df[col] = self.label_encoders[col].inverse_transform(
                        cf_df[col].clip(0, len(self.label_encoders[col].classes_) - 1)
                    )

            raw_cfs = cf_df.to_dict('records')

        # Decode query instance
        query_dict = query_instance.iloc[0].to_dict()
        for col in self.categorical_features:
            if col in self.label_encoders:
                val = int(round(query_dict[col]))
                val = max(0, min(val, len(self.label_encoders[col].classes_) - 1))
                query_dict[col] = self.label_encoders[col].inverse_transform([val])[0]

        # Filter counterfactuals: only keep feature columns with actual changes
        filtered_cfs = self._filter_meaningful_cfs(query_dict, raw_cfs)

        return query_dict, filtered_cfs

    def _filter_meaningful_cfs(self, query_dict, raw_cfs):
        """
        Filter counterfactuals to be meaningful:
        1. Remove non-feature columns (IDs, metadata, target)
        2. Only keep features that actually changed
        3. Remove CFs with no real changes
        
        Returns: List of meaningful CFs (dict format)
        """
        filtered_cfs = []
        
        for cf in raw_cfs:
            # Create a copy with only feature columns
            cf_cleaned = {}
            has_changes = False
            
            for feat in self.feature_names:
                if feat not in cf or feat == self.target:
                    continue
                
                # Get original value
                orig_val = query_dict.get(feat)
                cf_val = cf.get(feat)
                
                if orig_val is None or cf_val is None:
                    continue
                
                # Add to cleaned CF (always include feature columns)
                cf_cleaned[feat] = cf_val
                
                # Check if this feature changed
                try:
                    if float(orig_val) != float(cf_val):
                        has_changes = True
                except (ValueError, TypeError):
                    if str(orig_val) != str(cf_val):
                        has_changes = True
            
            # Only add if there are actual changes
            if has_changes and cf_cleaned:
                filtered_cfs.append(cf_cleaned)
        
        return filtered_cfs

    # ---- Step 4: Audit Counterfactuals ----
    def audit_counterfactuals(self, query_dict, raw_cfs, immutable_features, directional_rules=None):
        """
        Filter counterfactuals based on causal rules.
        
        immutable_features: list of feature names that cannot change
        directional_rules: dict of {feature: 'increase_only' or 'decrease_only'}
        """
        if directional_rules is None:
            directional_rules = {}

        valid_cfs = []
        invalid_cfs = []

        for cf in raw_cfs:
            is_valid = True
            reason = ""

            for feat in self.feature_names:
                if feat not in cf:
                    continue

                original_val = query_dict.get(feat)
                cf_val = cf.get(feat)

                # Skip if no change
                if str(original_val) == str(cf_val):
                    continue

                # Check immutability
                if feat in immutable_features:
                    is_valid = False
                    reason = f"Feature '{feat}' is immutable. Suggested: {original_val} → {cf_val}"
                    break

                # Check directional constraints
                if feat in directional_rules:
                    try:
                        orig_num = float(original_val)
                        cf_num = float(cf_val)
                        if directional_rules[feat] == 'increase_only' and cf_num < orig_num:
                            is_valid = False
                            reason = f"Feature '{feat}' can only increase. Suggested: {original_val} → {cf_val}"
                            break
                        elif directional_rules[feat] == 'decrease_only' and cf_num > orig_num:
                            is_valid = False
                            reason = f"Feature '{feat}' can only decrease. Suggested: {original_val} → {cf_val}"
                            break
                    except (ValueError, TypeError):
                        pass

            if is_valid:
                valid_cfs.append(cf)
            else:
                invalid_cfs.append({"suggestion": cf, "reason": reason})

        return valid_cfs, invalid_cfs

    # ---- Helpers ----
    def get_test_samples(self, n=20):
        """Return first n test samples for selection."""
        samples = self.X_test.head(n).copy()
        # Decode for readability
        for col in self.categorical_features:
            if col in self.label_encoders and col in samples.columns:
                samples[col] = samples[col].round().astype(int)
                samples[col] = self.label_encoders[col].inverse_transform(
                    samples[col].clip(0, len(self.label_encoders[col].classes_) - 1)
                )
        return samples

    def get_target_classes(self):
        """Return unique target classes."""
        if self.target in self.label_encoders:
            return list(self.label_encoders[self.target].classes_)
        return sorted(self.df[self.target].unique().tolist())

    def get_predicted_class(self, query_index):
        """Get the model's prediction for a query instance."""
        query = self.X_test.iloc[[query_index]]
        # FIXED: Use predict_proba instead of predict
        pred_proba = self.model.predict_proba(query)[0]
        pred = np.argmax(pred_proba)
        
        if self.target in self.label_encoders:
            return self.label_encoders[self.target].inverse_transform([int(pred)])[0]
        return pred
    
    def get_predicted_proba(self, query_index):
        """Get prediction probability for a query instance"""
        query = self.X_test.iloc[[query_index]]
        return self.model.predict_proba(query)[0]
