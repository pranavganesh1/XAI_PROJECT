#!/usr/bin/env python3
"""
Test script demonstrating the auto-cleaning pipeline
Shows before/after on a messy CSV
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from flask_app import auto_clean_dataframe

# Create a messy CSV for testing
print("=" * 70)
print("🧪 CREATING MESSY TEST DATA")
print("=" * 70)

# Create messy data
data = {
    'age': [25, 30, "35", 40, 25, 50, 99999, 25, None, 45] * 10,  # Duplicates, string, outlier, missing
    'income': [30000, 45000, 60000, "50000", 45000, 85000, -5000, 45000, 52000, None] * 10,  # String, outlier, negative, missing
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', None, 'D', 'B'] * 10,  # Missing
    'score': [0.5, 0.6, 0.7, 0.6, 0.5, 0.8, 0.9, 0.6, 0.5, None] * 10,  # Missing
}

df_messy = pd.DataFrame(data)

# Add exact duplicates
df_messy = pd.concat([df_messy, df_messy.head(10)], ignore_index=True)

print(f"\n📊 MESSY DATA CREATED:")
print(f"   Rows: {len(df_messy)}")
print(f"   Columns: {list(df_messy.columns)}")
print(f"   Total Nulls: {df_messy.isnull().sum().sum()}")
print(f"\nFirst 5 rows:")
print(df_messy.head(10).to_string())

# Test 1: With RAW mode
print("\n" + "=" * 70)
print("TEST 1: RAW MODE (no cleaning)")
print("=" * 70)
df_raw = auto_clean_dataframe(df_messy, clean_mode='raw')
print(f"Result: {len(df_raw)} rows, {len(df_raw.columns)} cols")

# Test 2: With AUTO-CLEAN mode
print("\n" + "=" * 70)
print("TEST 2: AUTO-CLEAN MODE (full cleaning)")
print("=" * 70)
df_clean = auto_clean_dataframe(df_messy, clean_mode='auto')
print(f"Result: {len(df_clean)} rows, {len(df_clean.columns)} cols")
print(f"\nFirst 5 rows after cleaning:")
print(df_clean.head(10).to_string())
print(f"\nData types after cleaning:")
print(df_clean.dtypes)

# Comparison
print("\n" + "=" * 70)
print("📊 COMPARISON: RAW vs CLEANED")
print("=" * 70)
print(f"Original:  {len(df_messy):4d} rows | Nulls: {df_messy.isnull().sum().sum():4d}")
print(f"Raw mode:  {len(df_raw):4d} rows | Nulls: {df_raw.isnull().sum().sum():4d}")
print(f"Cleaned:   {len(df_clean):4d} rows | Nulls: {df_clean.isnull().sum().sum():4d}")
print(f"\n✅ Rows removed: {len(df_messy) - len(df_clean)} (duplicates + outliers)")
print(f"✅ Missing values filled: {df_messy.isnull().sum().sum()}")

# Save test files
df_messy.to_csv('test_messy.csv', index=False)
df_clean.to_csv('test_cleaned.csv', index=False)
print(f"\n💾 Test files saved:")
print(f"   test_messy.csv (original)")
print(f"   test_cleaned.csv (after cleaning)")
print("\nTry uploading these to http://localhost:5000 with toggle ON/OFF!")

print("\n" + "=" * 70)
print("✅ TEST COMPLETE")
print("=" * 70)
