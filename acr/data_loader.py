"""
Data loader for ACR-PS experiments.
Supports: Adult, Diabetes, German Credit, Bank Marketing, COMPAS
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def get_adult_data(test_size=0.3):
    """
    Load Adult Census dataset.
    Target: income (binary: >50K = 1, <=50K = 0)
    """
    # Assumes adult.csv exists in data/ or is downloaded
    try:
        df = pd.read_csv('data/adult.csv')
    except FileNotFoundError:
        # Fallback: fetch from UCI
        from urllib.request import urlopen
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df = pd.read_csv(url, header=None)
        cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']
        df.columns = cols
    
    # Binary target
    df['income'] = (df['income'].str.strip() == '>50K').astype(int)
    return df


def get_diabetes_data(test_size=0.3):
    """
    Load Diabetes dataset.
    Target: Outcome (binary: 1 = positive, 0 = negative)
    """
    try:
        df = pd.read_csv('data/diabetes.csv')
    except FileNotFoundError:
        # Fallback: fetch from Kaggle or UCI
        from urllib.request import urlopen
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        df = pd.read_csv(url, header=None)
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                      'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    # Target is already binary (0/1)
    if 'Outcome' not in df.columns:
        df['Outcome'] = df.iloc[:, -1]
    
    return df


def get_german_credit_data():
    """
    Load German Credit dataset.
    Target: Good/Bad credit (1 = Good, 0 = Bad)
    
    Assumptions:
    - Data file is at data/german_credit_data.xls or raw format
    - Last column is the target (1=Good, 2=Bad → convert to binary)
    """
    try:
        # Try loading from XLS (your local copy)
        df = pd.read_excel('data/german_credit_data.xls', header=None)
    except FileNotFoundError:
        try:
            # Fallback: raw text format from UCI
            df = pd.read_csv('data/german.data', header=None, sep=' ')
        except FileNotFoundError:
            # Last resort: fetch from UCI
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            df = pd.read_csv(url, header=None, sep=' ')
    
    # Target is last column: 1=Good, 2=Bad → convert to 1=Good, 0=Bad
    target = df.iloc[:, -1]
    df['credit_target'] = (target == 1).astype(int)
    df = df.drop(columns=[df.columns[-2]])  # Remove original target
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    return df


def get_bank_marketing_data():
    """
    Load Bank Marketing dataset.
    Target: y (binary: 1 = client subscribed, 0 = did not subscribe)
    
    Assumptions:
    - Data file is at data/bank.xls or bank-additional-full.csv
    - Target column is 'y' (yes/no → convert to binary)
    """
    try:
        # Try loading from XLS (your local copy)
        df = pd.read_excel('data/bank.xls')
    except FileNotFoundError:
        try:
            # Fallback: CSV format
            df = pd.read_csv('data/bank-additional-full.csv', sep=';')
        except FileNotFoundError:
            # Last resort: fetch from UCI
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
            df = pd.read_csv(url, sep=';')
    
    # Convert target 'yes'/'no' to binary
    if 'y' in df.columns:
        df['subscription_target'] = (df['y'] == 'yes').astype(int)
        df = df.drop(columns=['y'])
    else:
        # If already numeric, use as-is
        df['subscription_target'] = df.iloc[:, -1].astype(int)
    
    # Handle missing values (common in this dataset)
    df = df.fillna(df.mean(numeric_only=True))
    
    return df


def get_compas_data():
    """
    Load COMPAS dataset.
    Target: two_year_recidivism (binary: 1 = recidivated, 0 = no recidivism)
    
    Assumptions:
    - Data file is at data/compas-scores-two-years.csv or fetch from GitHub
    - We use 'two_year_recidivism' as target (if available, else reconstruct)
    - Focus on defendant-level features (exclude case identifiers)
    """
    try:
        # Try local file
        df = pd.read_csv('data/compas-scores-two-years.csv')
    except FileNotFoundError:
        # Fetch from ProPublica's GitHub
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        df = pd.read_csv(url)
    
    # Identify and create binary recidivism target
    if 'two_year_recidivism' in df.columns:
        df['recidivism_target'] = df['two_year_recidivism'].astype(int)
    elif 'is_recidivate' in df.columns:
        df['recidivism_target'] = df['is_recidivate'].astype(int)
    else:
        # Fallback: use 'c_jail_in' and 'c_jail_out' to infer recidivism
        # (simplified; real COMPAS analysis is more nuanced)
        df['recidivism_target'] = 0  # Placeholder
    
    # Drop case identifiers and non-features
    drop_cols = [col for col in df.columns if col in [
        'id', 'case_number', 'screening_date', 'compas_screening_date',
        'dob', 'c_case_number', 'c_offense_date', 'c_arrest_date',
        'c_jail_in', 'c_jail_out', 'r_case_number', 'r_offense_date',
        'r_arrest_date', 'r_jail_in', 'r_jail_out'
    ]]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    return df


def preprocess_dataset(df, dataset_name, categorical_threshold=10):
    """
    Minimal preprocessing: handle categoricals, missing values.
    
    Args:
        df: DataFrame
        dataset_name: str, for dataset-specific logic
        categorical_threshold: int, columns with <N unique values treated as categorical
    
    Returns:
        Processed DataFrame with separated features/target
    """
    df = df.copy()
    
    # Identify target column based on dataset
    target_col = {
        'adult': 'income',
        'diabetes': 'Outcome',
        'german_credit': 'credit_target',
        'bank_marketing': 'subscription_target',
        'compas': 'recidivism_target',
    }.get(dataset_name, df.columns[-1])
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {dataset_name}")
    
    # Separate target
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    
    # One-hot encode categoricals (optional, for now keep as-is for rule detection)
    return X, y, target_col


def load_and_prepare(dataset_name, local_data_path='data/'):
    """
    Main interface: load dataset, preprocess, return X, y.
    
    Args:
        dataset_name: str, one of ['adult', 'diabetes', 'german_credit', 'bank_marketing', 'compas']
        local_data_path: str, path to local data files
    
    Returns:
        X: DataFrame of features
        y: Series of binary targets
        target_col: str, name of target column
    """
    loaders = {
        'adult': get_adult_data,
        'diabetes': get_diabetes_data,
        'german_credit': get_german_credit_data,
        'bank_marketing': get_bank_marketing_data,
        'compas': get_compas_data,
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    
    df = loaders[dataset_name]()
    X, y, target_col = preprocess_dataset(df, dataset_name)
    
    return X, y, target_col


if __name__ == '__main__':
    # Quick test
    for ds_name in ['adult', 'diabetes', 'german_credit', 'bank_marketing', 'compas']:
        try:
            X, y, target = load_and_prepare(ds_name)
            print(f"{ds_name}: X={X.shape}, y={y.shape}, target={target}")
        except Exception as e:
            print(f"{ds_name}: Error - {e}")