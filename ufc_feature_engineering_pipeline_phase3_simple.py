#!/usr/bin/env python3
"""
UFC Feature Engineering Pipeline - Phase 3 (Simplified)
Phase 3: Feature Engineering Integration with ELO Features
Simplified approach using ELO dataset as base
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_elo_dataset():
    """Load the ELO dataset created in Phase 2"""
    print("Loading ELO dataset...")
    df = pd.read_csv("dataUFC/ufc_elo_dataset.csv")
    print(f"ELO dataset shape: {df.shape}")
    return df

def add_missing_features_from_engineered(elo_df):
    """
    Add any missing features from the engineered dataset to the ELO dataset
    """
    print("Checking for additional features from engineered dataset...")
    
    try:
        engineered_df = pd.read_csv("dataUFC/ufc_engineered.csv")
        print(f"Engineered dataset shape: {engineered_df.shape}")
        
        # Identify features in engineered dataset that are not in ELO dataset
        elo_features = set(elo_df.columns)
        eng_features = set(engineered_df.columns)
        
        missing_features = eng_features - elo_features
        print(f"Features in engineered dataset not in ELO dataset: {len(missing_features)}")
        
        if missing_features:
            print("Missing features:")
            for feature in sorted(missing_features):
                print(f"  - {feature}")
            
            # For now, we'll focus on the ELO dataset as our primary source
            # since it contains all the ELO features we need
            print("\nUsing ELO dataset as primary source (contains all ELO features)")
        else:
            print("✓ All features from engineered dataset are already in ELO dataset")
            
    except FileNotFoundError:
        print("⚠️ Engineered dataset not found. Using ELO dataset only.")
    
    return elo_df

def validate_elo_features(elo_df):
    """Validate that all ELO features are present and complete"""
    print("Validating ELO features...")
    
    # Core ELO features
    core_elo_features = [
        'ELO_DIFF', 'ELO_WEIGHTCLASS_DIFF', 'N_FIGHTS_DIFF',
        'H2H_DIFF', 'H2H_WEIGHTCLASS_DIFF'
    ]
    
    # K-value features (rolling statistics)
    k_value_features = []
    for k in [3, 5, 10, 25, 50, 100, 200]:
        k_value_features.extend([
            f'WIN_LAST_{k}_DIFF',
            f'ELO_GRAD_LAST_{k}_DIFF'
        ])
    
    # UFC-specific K-value features
    ufc_k_features = []
    for k in [3, 5, 10, 25, 50, 100, 200]:
        ufc_k_features.extend([
            f'STRIKE_ACC_LAST_{k}_DIFF',
            f'TD_ACC_LAST_{k}_DIFF',
            f'KO_RATE_LAST_{k}_DIFF',
            f'SUB_RATE_LAST_{k}_DIFF'
        ])
    
    all_elo_features = core_elo_features + k_value_features + ufc_k_features
    
    # Check which features are present
    available_elo_features = [f for f in all_elo_features if f in elo_df.columns]
    missing_elo_features = [f for f in all_elo_features if f not in elo_df.columns]
    
    print(f"Available ELO features: {len(available_elo_features)}")
    print(f"Missing ELO features: {len(missing_elo_features)}")
    
    if missing_elo_features:
        print("Missing ELO features:")
        for feature in missing_elo_features:
            print(f"  - {feature}")
    
    # Check for missing values in ELO features
    if available_elo_features:
        elo_missing = elo_df[available_elo_features].isnull().sum()
        print(f"\nMissing values in ELO features:")
        missing_count = 0
        for feature, missing in elo_missing.items():
            if missing > 0:
                print(f"  {feature}: {missing} ({missing/len(elo_df)*100:.1f}%)")
                missing_count += missing
            else:
                print(f"  {feature}: ✅ Complete")
        
        if missing_count == 0:
            print("🎉 All ELO features have complete data!")
    
    return available_elo_features

def create_feature_summary(elo_df, elo_features):
    """Create a comprehensive feature summary"""
    print("Creating feature summary...")
    
    # Categorize all features
    feature_categories = {
        'ELO Core': ['ELO_DIFF', 'ELO_WEIGHTCLASS_DIFF', 'N_FIGHTS_DIFF', 'H2H_DIFF', 'H2H_WEIGHTCLASS_DIFF'],
        'ELO Rolling (Win/Loss)': [f for f in elo_features if 'WIN_LAST' in f],
        'ELO Rolling (Gradient)': [f for f in elo_features if 'ELO_GRAD' in f],
        'ELO Rolling (UFC Stats)': [f for f in elo_features if any(x in f for x in ['STRIKE', 'TD_ACC', 'KO_RATE', 'SUB_RATE'])],
        'Identifiers': ['RedFighterID', 'BlueFighterID', 'RedFighter', 'BlueFighter', 'Date', 'WeightClass', 'RESULT']
    }
    
    print(f"\n=== Feature Categories ===")
    total_features = 0
    for category, features in feature_categories.items():
        available_features = [f for f in features if f in elo_df.columns]
        if available_features:
            print(f"{category}: {len(available_features)} features")
            total_features += len(available_features)
    
    # Count remaining features
    all_categorized = []
    for features in feature_categories.values():
        all_categorized.extend(features)
    
    remaining_features = [f for f in elo_df.columns if f not in all_categorized]
    if remaining_features:
        print(f"Other: {len(remaining_features)} features")
        total_features += len(remaining_features)
    
    print(f"\nTotal features: {total_features}")
    print(f"Dataset shape: {elo_df.shape}")
    
    return feature_categories

def save_final_dataset(elo_df, output_path="dataUFC/ufc_final_with_elo.csv"):
    """Save the final dataset with ELO features"""
    print(f"Saving final dataset with ELO features...")
    elo_df.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}")
    return output_path

def main():
    """Main Phase 3 feature engineering function"""
    print("=== UFC Feature Engineering Pipeline - Phase 3 (Simplified) ===")
    print("Using ELO dataset as primary source for feature integration\n")
    
    # Load ELO dataset
    elo_df = load_elo_dataset()
    
    # Check for additional features from engineered dataset
    elo_df = add_missing_features_from_engineered(elo_df)
    
    # Validate ELO features
    elo_features = validate_elo_features(elo_df)
    
    # Create feature summary
    feature_categories = create_feature_summary(elo_df, elo_features)
    
    # Save final dataset
    output_path = save_final_dataset(elo_df)
    
    # Final summary
    print(f"\n=== Phase 3 Summary ===")
    print(f"ELO dataset: {elo_df.shape}")
    print(f"ELO features available: {len(elo_features)}")
    print(f"Total features: {elo_df.shape[1]}")
    
    # Show key ELO features
    print(f"\nKey ELO features available:")
    key_elo_features = ['ELO_DIFF', 'ELO_WEIGHTCLASS_DIFF', 'N_FIGHTS_DIFF', 'H2H_DIFF', 'H2H_WEIGHTCLASS_DIFF']
    for feature in key_elo_features:
        if feature in elo_df.columns:
            print(f"  ✅ {feature}")
        else:
            print(f"  ❌ {feature} (missing)")
    
    # Show target distribution
    if 'RESULT' in elo_df.columns:
        target_dist = elo_df['RESULT'].value_counts()
        print(f"\nTarget distribution:")
        print(f"  Fighter1 wins (1): {target_dist[1]} ({target_dist[1]/len(elo_df)*100:.1f}%)")
        print(f"  Fighter2 wins (0): {target_dist[0]} ({target_dist[0]/len(elo_df)*100:.1f}%)")
    
    # Show ELO feature statistics
    print(f"\nELO feature statistics:")
    if 'ELO_DIFF' in elo_df.columns:
        elo_stats = elo_df['ELO_DIFF'].describe()
        print(f"  ELO_DIFF: {elo_stats['min']:.1f} to {elo_stats['max']:.1f} (mean: {elo_stats['mean']:.1f})")
    
    if 'ELO_WEIGHTCLASS_DIFF' in elo_df.columns:
        wc_elo_stats = elo_df['ELO_WEIGHTCLASS_DIFF'].describe()
        print(f"  ELO_WEIGHTCLASS_DIFF: {wc_elo_stats['min']:.1f} to {wc_elo_stats['max']:.1f} (mean: {wc_elo_stats['mean']:.1f})")
    
    print(f"\nPhase 3 Feature Engineering Integration Complete!")
    print(f"Ready to proceed to Phase 4: Model Training with ELO Features")

if __name__ == "__main__":
    main()
