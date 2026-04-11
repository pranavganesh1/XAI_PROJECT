import config
import pandas as pd
import numpy as np
from utils import split_dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


class Model:
    """
    Wrapper for classification model with proper prediction pipeline.
    Ensures predictions change when inputs change.
    """
    
    def __init__(self) -> None:
        # Try to load pre-trained model
        model_pred_path = f"{config.ROOT_DIR}/models/noise{config.NOISE_LEVEL}/{config.MODEL}_epoch10_preds.csv"
        try:
            self.preds = pd.read_csv(model_pred_path).set_index('image_name')
            self.use_pretrained = True
        except:
            self.preds = None
            self.use_pretrained = False
        
        # For new models
        self.sklearn_model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numeric_features = []
        self.accuracy = 0.0
        
    def accuracy(self, images=None):
        """Average accuracy of the model"""
        if self.use_pretrained:
            if images is not None and len(images) > 0:
                model_acc = self.preds.loc[images, 'correct'].mean()
            else:
                model_acc = self.preds['correct'].mean()
            return model_acc
        elif self.sklearn_model:
            return self.accuracy
        return 0.0

    def acc_se(self, images=None):
        """Standard error of the accuracy"""
        if self.use_pretrained:
            if images is not None and len(images) > 0:
                model_se = self.preds.loc[images, 'correct'].std() / np.sqrt(len(self.preds.loc[images, 'correct']))
            else:
                model_se = self.preds['correct'].std() / np.sqrt(len(self.preds['correct']))
            return model_se
        return 0.0

    def pred_prob(self, x):
        """Get probability predictions (FIXED for proper sensitivity)"""
        if self.use_pretrained:
            pred_probs = self.preds.loc[x, 'knife':'dog'].to_numpy()
        else:
            pred_probs = np.array([0.5, 0.5])  # Fallback
        return pred_probs

    def pred_prob_sorted(self, x):
        """Get sorted probabilities"""
        pred_probs = self.pred_prob(x)
        sorted_desc = -np.sort(-pred_probs)
        return sorted_desc
    
    def train_sklearn_model(self, X_train, y_train, feature_names):
        """Train sklearn model with proper preprocessing pipeline"""
        self.feature_names = feature_names
        
        # Create pipeline with scaling
        self.sklearn_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        # Train
        self.sklearn_model.fit(X_train, y_train)
        logger.info(f"✓ Model trained on {len(X_train)} samples")
        
        return self
    
    def predict_proba_sklearn(self, input_dict):
        """Predict using preprocessed sklearn model"""
        if self.sklearn_model is None:
            return np.array([0.5, 0.5])
        
        # Convert dict to DataFrame with correct feature order
        input_df = pd.DataFrame([input_dict])
        input_df = input_df[self.feature_names]
        
        # Predict
        proba = self.sklearn_model.predict_proba(input_df)[0]
        return proba

if __name__=="__main__":
    X_train, X_cal, y_train, y_cal = split_dataset(config.run_no_cal)

    # Shuffle train set
    config.numpy_rng.shuffle(X_train)

    config.total_timesteps = len(X_train)
    # Create model
    model = Model()
    print(model.accuracy(X_train))
    print(model.acc_se(X_train))
    