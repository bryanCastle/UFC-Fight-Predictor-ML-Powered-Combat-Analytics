#!/usr/bin/env python3
"""
UFC Data Quality Assessment Script
Phase 1.2: Data Quality Assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime

def main():
    print("=== UFC Data Quality Assessment ===\n")
    
    # Load datasets
    print("Loading datasets...")
    ufc_master = pd.read_csv("dataUFC/ufc-master.csv")
    ufc_fight_stats = pd.read_csv("dataUFC/ufc_fight_stats.csv")
    
    print(f"UFC Master Dataset: {ufc_master.shape}")
    print(f"UFC Fight Stats Dataset: {ufc_fight_stats.shape}\n")
    
    # 1. Missing Values Analysis
    print("=== MISSING VALUES ANALYSIS ===")
    
    # UFC Master missing values
    missing_master = ufc_master.isnull().sum()
    missing_master_percent = (missing_master / len(ufc_master)) * 100
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_master,
        'Missing_Percent': missing_master_percent
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    print("UFC Master Dataset - Columns with missing values:")
    print(missing_summary.head(10))
    print(f"Total columns with missing values: {len(missing_summary)}\n")
    
    # UFC Fight Stats missing values
    missing_stats = ufc_fight_stats.isnull().sum()
    missing_stats_percent = (missing_stats / len(ufc_fight_stats)) * 100
    missing_stats_summary = pd.DataFrame({
        'Missing_Count': missing_stats,
        'Missing_Percent': missing_stats_percent
    })
    missing_stats_summary = missing_stats_summary[missing_stats_summary['Missing_Count'] > 0]
    
    print("UFC Fight Stats Dataset - Columns with missing values:")
    print(missing_stats_summary)
    print(f"Total columns with missing values: {len(missing_stats_summary)}\n")
    
    # 2. Key Column Analysis
    print("=== KEY COLUMN ANALYSIS ===")
    
    # Winner analysis
    print("Winner values:")
    print(ufc_master['Winner'].value_counts())
    print()
    
    # Weight class analysis
    print("Weight Class distribution:")
    print(ufc_master['WeightClass'].value_counts())
    print()
    
    # Gender analysis
    print("Gender distribution:")
    print(ufc_master['Gender'].value_counts())
    print()
    
    # 3. Temporal Coverage
    print("=== TEMPORAL COVERAGE ===")
    ufc_master['Date'] = pd.to_datetime(ufc_master['Date'], errors='coerce')
    
    print(f"Date range:")
    print(f"  Earliest: {ufc_master['Date'].min()}")
    print(f"  Latest: {ufc_master['Date'].max()}")
    print(f"  Total days: {(ufc_master['Date'].max() - ufc_master['Date'].min()).days}")
    
    # Fights per year
    fights_per_year = ufc_master.groupby(ufc_master['Date'].dt.year).size()
    print(f"\nFights per year (last 5 years):")
    print(fights_per_year.tail())
    print()
    
    # 4. Duplicate Analysis
    print("=== DUPLICATE ANALYSIS ===")
    
    # Exact duplicates
    exact_duplicates = ufc_master.duplicated().sum()
    print(f"Exact duplicate rows: {exact_duplicates}")
    
    # Duplicate fights (same fighters, same date)
    fight_duplicates = ufc_master.duplicated(subset=['RedFighter', 'BlueFighter', 'Date']).sum()
    print(f"Duplicate fights (same fighters, same date): {fight_duplicates}")
    
    # Rematches (same fighters, different dates)
    fighter_pairs = ufc_master.groupby(['RedFighter', 'BlueFighter']).size()
    rematches = (fighter_pairs > 1).sum()
    print(f"Fighter pairs with multiple fights (rematches): {rematches}")
    print()
    
    # 5. Numerical Data Quality
    print("=== NUMERICAL DATA QUALITY ===")
    
    numerical_columns = ['RedAge', 'BlueAge', 'RedHeightCms', 'BlueHeightCms', 
                        'RedReachCms', 'BlueReachCms', 'RedWeightLbs', 'BlueWeightLbs']
    
    for col in numerical_columns:
        if col in ufc_master.columns:
            print(f"\n{col}:")
            print(f"  Min: {ufc_master[col].min()}")
            print(f"  Max: {ufc_master[col].max()}")
            print(f"  Mean: {ufc_master[col].mean():.2f}")
            print(f"  Missing: {ufc_master[col].isnull().sum()}")
            
            # Check for unrealistic values
            if 'Age' in col:
                unrealistic = ((ufc_master[col] < 16) | (ufc_master[col] > 60)).sum()
                print(f"  Unrealistic ages (<16 or >60): {unrealistic}")
            elif 'Height' in col:
                unrealistic = ((ufc_master[col] < 150) | (ufc_master[col] > 220)).sum()
                print(f"  Unrealistic heights (<150cm or >220cm): {unrealistic}")
            elif 'Weight' in col:
                unrealistic = ((ufc_master[col] < 100) | (ufc_master[col] > 400)).sum()
                print(f"  Unrealistic weights (<100lbs or >400lbs): {unrealistic}")
    
    # 6. Dataset Consistency
    print("\n=== DATASET CONSISTENCY ===")
    
    # Check fighter overlap
    master_fighters = set(ufc_master['RedFighter'].dropna()) | set(ufc_master['BlueFighter'].dropna())
    stats_fighters = set(ufc_fight_stats['fighter_1'].dropna()) | set(ufc_fight_stats['fighter_2'].dropna())
    
    print(f"Fighters in master dataset: {len(master_fighters)}")
    print(f"Fighters in fight stats dataset: {len(stats_fighters)}")
    print(f"Fighters in both datasets: {len(master_fighters & stats_fighters)}")
    print(f"Fighters only in master: {len(master_fighters - stats_fighters)}")
    print(f"Fighters only in fight stats: {len(stats_fighters - master_fighters)}")
    
    # 7. Data Quality Summary
    print("\n=== DATA QUALITY SUMMARY ===")
    print("\nIssues Found:")
    print("1. Missing values in various columns (especially rankings and betting odds)")
    print("2. Different naming conventions between datasets")
    print("3. Mixed data formats (ratios, time formats)")
    print("4. Limited overlap between master and fight stats datasets")
    print("5. Some unrealistic values in physical attributes")
    
    print("\nRecommendations:")
    print("1. Focus on ufc-master.csv as primary dataset (more comprehensive)")
    print("2. Handle missing values appropriately (imputation or removal)")
    print("3. Standardize fighter names across datasets")
    print("4. Convert ratio formats to numerical values")
    print("5. Validate physical attribute ranges")
    print("6. Create consistent fighter IDs")
    print("7. Use fight stats dataset for additional features when available")

if __name__ == "__main__":
    main()
