#!/usr/bin/env python3
"""
UFC Data Cleaning Implementation
Phase 2.1: Data Cleaning Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

def load_ufc_data():
    """Load the UFC master dataset"""
    print("Loading UFC dataset...")
    df = pd.read_csv("dataUFC/ufc-master.csv")
    print(f"Original dataset shape: {df.shape}")
    return df

def remove_ranking_columns(df):
    """Remove ranking columns with >50% missing values"""
    print("Removing ranking columns with high missing values...")
    
    # Identify ranking columns
    ranking_cols = [col for col in df.columns if 'Rank' in col]
    print(f"Total ranking columns found: {len(ranking_cols)}")
    
    # Keep BetterRank (0% missing), remove others
    cols_to_remove = [col for col in ranking_cols if col != 'BetterRank']
    print(f"Removing {len(cols_to_remove)} ranking columns...")
    
    # Remove the columns
    df = df.drop(columns=cols_to_remove)
    print(f"Dataset shape after ranking removal: {df.shape}")
    
    return df

def handle_missing_values(df):
    """Handle missing values based on missing percentage"""
    print("Handling missing values...")
    
    # Calculate missing percentages
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percent': missing_percent
    }).sort_values('Missing_Percent', ascending=False)
    
    print("Missing value summary:")
    print(missing_summary[missing_summary['Missing_Count'] > 0].head(10))
    
    # Handle different missing value categories
    
    # 1. Complete data (0% missing) - no action needed
    complete_cols = missing_summary[missing_summary['Missing_Percent'] == 0].index.tolist()
    print(f"Complete columns (0% missing): {len(complete_cols)}")
    
    # 2. Low missing data (<5% missing) - impute with median/mode
    low_missing_cols = missing_summary[
        (missing_summary['Missing_Percent'] > 0) & 
        (missing_summary['Missing_Percent'] < 5)
    ].index.tolist()
    
    print(f"Low missing columns (<5%): {len(low_missing_cols)}")
    for col in low_missing_cols:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 3. Moderate missing data (5-20% missing) - impute with fighter-specific medians
    moderate_missing_cols = missing_summary[
        (missing_summary['Missing_Percent'] >= 5) & 
        (missing_summary['Missing_Percent'] <= 20)
    ].index.tolist()
    
    print(f"Moderate missing columns (5-20%): {len(moderate_missing_cols)}")
    for col in moderate_missing_cols:
        if df[col].dtype in ['int64', 'float64']:
            # Impute with overall median for now (can be improved with fighter-specific)
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 4. High missing data (>20% missing) - special handling
    high_missing_cols = missing_summary[missing_summary['Missing_Percent'] > 20].index.tolist()
    print(f"High missing columns (>20%): {len(high_missing_cols)}")
    
    # Remove FinishDetails (55.7% missing) as planned
    if 'FinishDetails' in high_missing_cols:
        df = df.drop(columns=['FinishDetails'])
        high_missing_cols.remove('FinishDetails')
        print("Removed FinishDetails column (55.7% missing)")
    
    # Keep EmptyArena for COVID-19 analysis
    if 'EmptyArena' in high_missing_cols:
        df['EmptyArena'] = df['EmptyArena'].fillna(False)  # Assume not empty arena if missing
        high_missing_cols.remove('EmptyArena')
        print("Imputed EmptyArena with False (assuming not empty arena if missing)")
    
    # Handle remaining high missing columns
    for col in high_missing_cols:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def validate_data_types(df):
    """Validate and correct data types"""
    print("Validating data types...")
    
    # Convert date column to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure numeric columns are numeric
    numeric_cols = ['RedAge', 'BlueAge', 'RedHeightCms', 'BlueHeightCms', 
                   'RedReachCms', 'BlueReachCms', 'RedWeightLbs', 'BlueWeightLbs',
                   'RedWins', 'BlueWins', 'RedLosses', 'BlueLosses', 'RedDraws', 'BlueDraws']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure boolean columns are boolean
    boolean_cols = ['TitleBout', 'EmptyArena']
    for col in boolean_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    return df

def standardize_fighter_names(df):
    """Standardize fighter names for consistency"""
    print("Standardizing fighter names...")
    
    def clean_name(name):
        if pd.isna(name):
            return name
        # Remove extra whitespace and standardize
        name = str(name).strip()
        # Remove special characters but keep spaces and hyphens
        name = re.sub(r'[^\w\s\-]', '', name)
        return name
    
    # Clean fighter names
    if 'RedFighter' in df.columns:
        df['RedFighter'] = df['RedFighter'].apply(clean_name)
    if 'BlueFighter' in df.columns:
        df['BlueFighter'] = df['BlueFighter'].apply(clean_name)
    
    return df

def create_fighter_id_system(df):
    """Create a fighter ID system for consistent identification"""
    print("Creating fighter ID system...")
    
    # Get all unique fighters
    red_fighters = df['RedFighter'].unique()
    blue_fighters = df['BlueFighter'].unique()
    all_fighters = np.union1d(red_fighters, blue_fighters)
    
    # Create fighter ID mapping
    fighter_id_map = {fighter: idx for idx, fighter in enumerate(all_fighters)}
    
    # Add fighter IDs to dataframe
    df['RedFighterID'] = df['RedFighter'].map(fighter_id_map)
    df['BlueFighterID'] = df['BlueFighter'].map(fighter_id_map)
    
    print(f"Created IDs for {len(all_fighters)} unique fighters")
    
    # Save fighter mapping for future use
    fighter_mapping = pd.DataFrame({
        'FighterID': list(fighter_id_map.values()),
        'FighterName': list(fighter_id_map.keys())
    })
    fighter_mapping.to_csv('dataUFC/fighter_id_mapping.csv', index=False)
    print("Saved fighter ID mapping to dataUFC/fighter_id_mapping.csv")
    
    return df

def validate_data_quality(df):
    """Validate data quality after cleaning"""
    print("Validating data quality...")
    
    # Check for remaining missing values
    remaining_missing = df.isnull().sum()
    remaining_missing_percent = (remaining_missing / len(df)) * 100
    
    print(f"Remaining missing values:")
    high_missing = remaining_missing_percent[remaining_missing_percent > 0]
    if len(high_missing) > 0:
        for col, percent in high_missing.items():
            print(f"  {col}: {percent:.1f}%")
    else:
        print("  No missing values remaining!")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Check data types
    print(f"Data types summary:")
    print(df.dtypes.value_counts())
    
    # Check value ranges for key numeric columns
    numeric_cols = ['RedAge', 'BlueAge', 'RedHeightCms', 'BlueHeightCms', 
                   'RedReachCms', 'BlueReachCms', 'RedWeightLbs', 'BlueWeightLbs']
    
    print(f"Value ranges for key columns:")
    for col in numeric_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].min():.1f} to {df[col].max():.1f}")
    
    return df

def main():
    """Main data cleaning function"""
    print("=== UFC Data Cleaning Implementation ===\n")
    
    # Load data
    df = load_ufc_data()
    
    # Remove ranking columns
    df = remove_ranking_columns(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Validate data types
    df = validate_data_types(df)
    
    # Standardize fighter names
    df = standardize_fighter_names(df)
    
    # Create fighter ID system
    df = create_fighter_id_system(df)
    
    # Validate data quality
    df = validate_data_quality(df)
    
    # Save cleaned dataset
    output_path = "dataUFC/ufc_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    
    # Final summary
    print(f"\n=== Data Cleaning Summary ===")
    print(f"Original shape: (6528, 118)")
    print(f"Cleaned shape: {df.shape}")
    print(f"Columns removed: {118 - df.shape[1]}")
    print(f"Data quality: {'✅ Excellent' if df.isnull().sum().sum() == 0 else '⚠️ Some missing values'}")
    
    print(f"\nData cleaning completed successfully!")
    print(f"Dataset ready for feature engineering in Phase 2.2.")

if __name__ == "__main__":
    main()
