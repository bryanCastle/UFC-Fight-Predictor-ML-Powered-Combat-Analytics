#!/usr/bin/env python3
"""
UFC Feature Engineering Demonstration
Phase 1.3: Feature Engineering Planning Implementation
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data():
    """Load and perform initial data cleaning"""
    print("Loading UFC dataset...")
    df = pd.read_csv("dataUFC/ufc-master.csv")
    
    # Remove missing ranking columns (28 columns with >50% missing)
    ranking_cols = [col for col in df.columns if 'Rank' in col]
    cols_to_remove = [col for col in ranking_cols if col != 'BetterRank']
    df = df.drop(columns=cols_to_remove)
    
    print(f"Dataset shape after removing ranking columns: {df.shape}")
    return df

def create_physical_advantage_features(df):
    """Create physical advantage features"""
    print("Creating physical advantage features...")
    
    # Height advantage
    df['height_diff'] = df['RedHeightCms'] - df['BlueHeightCms']
    df['height_ratio'] = df['RedHeightCms'] / df['BlueHeightCms']
    
    # Reach advantage
    df['reach_diff'] = df['RedReachCms'] - df['BlueReachCms']
    df['reach_ratio'] = df['RedReachCms'] / df['BlueReachCms']
    
    # Weight advantage
    df['weight_diff'] = df['RedWeightLbs'] - df['BlueWeightLbs']
    df['weight_ratio'] = df['RedWeightLbs'] / df['BlueWeightLbs']
    
    # Age advantage
    df['age_diff'] = df['RedAge'] - df['BlueAge']
    df['age_ratio'] = df['RedAge'] / df['BlueAge']
    
    # Overall physical advantage score
    df['physical_advantage'] = (
        (df['height_diff'] * 0.3) + 
        (df['reach_diff'] * 0.4) + 
        (df['weight_diff'] * 0.2) + 
        (df['age_diff'] * 0.1)
    ) / 4
    
    return df

def create_experience_features(df):
    """Create experience and career features"""
    print("Creating experience features...")
    
    # Total fights
    df['red_total_fights'] = df['RedWins'] + df['RedLosses'] + df['RedDraws']
    df['blue_total_fights'] = df['BlueWins'] + df['BlueLosses'] + df['BlueDraws']
    df['experience_diff'] = df['red_total_fights'] - df['blue_total_fights']
    
    # Win rates
    df['red_win_rate'] = df['RedWins'] / (df['red_total_fights'] + 1)  # +1 to avoid division by zero
    df['blue_win_rate'] = df['BlueWins'] / (df['blue_total_fights'] + 1)
    df['win_rate_diff'] = df['red_win_rate'] - df['blue_win_rate']
    
    # Win method percentages
    df['red_ko_rate'] = df['RedWinsByKO'] / (df['RedWins'] + 1)
    df['blue_ko_rate'] = df['BlueWinsByKO'] / (df['BlueWins'] + 1)
    df['ko_rate_diff'] = df['red_ko_rate'] - df['blue_ko_rate']
    
    df['red_sub_rate'] = df['RedWinsBySubmission'] / (df['RedWins'] + 1)
    df['blue_sub_rate'] = df['BlueWinsBySubmission'] / (df['BlueWins'] + 1)
    df['sub_rate_diff'] = df['red_sub_rate'] - df['blue_sub_rate']
    
    return df

def create_fight_context_features(df):
    """Create fight context and categorical features"""
    print("Creating fight context features...")
    
    # Weight class encoding
    weight_class_mapping = {
        'Flyweight': 1, 'Bantamweight': 2, 'Featherweight': 3,
        'Lightweight': 4, 'Welterweight': 5, 'Middleweight': 6,
        'Light Heavyweight': 7, 'Heavyweight': 8,
        'Women\'s Strawweight': 9, 'Women\'s Flyweight': 10,
        'Women\'s Bantamweight': 11, 'Women\'s Featherweight': 12,
        'Catch Weight': 13
    }
    df['weight_class_encoded'] = df['WeightClass'].map(weight_class_mapping)
    
    # Gender encoding
    df['gender_encoded'] = df['Gender'].map({'MALE': 1, 'FEMALE': 0})
    
    # Title bout encoding
    df['title_bout_encoded'] = df['TitleBout'].map({True: 1, False: 0})
    
    # BetterRank encoding
    better_rank_mapping = {'Red': 1, 'Blue': -1, 'neither': 0}
    df['better_rank_encoded'] = df['BetterRank'].map(better_rank_mapping)
    
    return df

def create_style_features(df):
    """Create style matchup features"""
    print("Creating style features...")
    
    # Stance matchup
    def get_stance_matchup(row):
        red_stance = row['RedStance']
        blue_stance = row['BlueStance']
        
        if pd.isna(red_stance) or pd.isna(blue_stance):
            return 0
        
        if red_stance == blue_stance:
            return 0  # Same stance
        elif (red_stance == 'Southpaw' and blue_stance == 'Orthodox') or \
             (red_stance == 'Orthodox' and blue_stance == 'Southpaw'):
            return 1  # Southpaw advantage
        else:
            return 0  # Other combinations
    
    df['stance_matchup'] = df.apply(get_stance_matchup, axis=1)
    
    # Experience vs Youth
    df['red_experienced'] = ((df['red_total_fights'] > df['blue_total_fights']) & 
                            (df['RedAge'] > df['BlueAge'])).astype(int)
    df['blue_experienced'] = ((df['blue_total_fights'] > df['red_total_fights']) & 
                             (df['BlueAge'] > df['RedAge'])).astype(int)
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("Handling missing values...")
    
    # Performance metrics (impute with median)
    performance_cols = ['RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 
                       'RedAvgSigStrPct', 'BlueAvgSigStrPct',
                       'RedAvgSubAtt', 'BlueAvgSubAtt',
                       'RedAvgTDLanded', 'BlueAvgTDLanded',
                       'RedAvgTDPct', 'BlueAvgTDPct']
    
    for col in performance_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Betting odds (impute with median)
    betting_cols = ['RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue']
    for col in betting_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Fighter stance (impute with mode)
    if 'BlueStance' in df.columns:
        df['BlueStance'] = df['BlueStance'].fillna(df['BlueStance'].mode()[0])
    
    return df

def implement_randomization(df):
    """Implement fighter randomization (similar to tennis model)"""
    print("Implementing fighter randomization...")
    
    # Generate 50/50 randomization mask
    np.random.seed(42)  # For reproducibility
    mask = np.random.rand(len(df)) < 0.5
    
    # Identify fighter columns
    red_cols = [col for col in df.columns if col.startswith('Red')]
    blue_cols = [col for col in df.columns if col.startswith('Blue')]
    
    # Create RESULT column (1 = Fighter1 wins, 0 = Fighter2 wins)
    df['RESULT'] = np.where(mask, 0, 1)
    
    # Swap fighters where mask is True
    df.loc[mask, red_cols], df.loc[mask, blue_cols] = df.loc[mask, blue_cols].values, df.loc[mask, red_cols].values
    
    # Recalculate derived features after randomization
    df = create_physical_advantage_features(df)
    df = create_experience_features(df)
    df = create_style_features(df)
    
    return df

def main():
    """Main function to demonstrate feature engineering"""
    print("=== UFC Feature Engineering Demonstration ===\n")
    
    # Load and clean data
    df = load_and_clean_data()
    
    # Create features
    df = create_physical_advantage_features(df)
    df = create_experience_features(df)
    df = create_fight_context_features(df)
    df = create_style_features(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Implement randomization
    df = implement_randomization(df)
    
    # Show results
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Total features created: {len(df.columns)}")
    
    # Show sample of new features
    new_features = ['height_diff', 'reach_diff', 'weight_diff', 'age_diff', 
                   'physical_advantage', 'experience_diff', 'win_rate_diff',
                   'ko_rate_diff', 'sub_rate_diff', 'weight_class_encoded',
                   'gender_encoded', 'title_bout_encoded', 'better_rank_encoded',
                   'stance_matchup', 'red_experienced', 'blue_experienced', 'RESULT']
    
    print(f"\nSample of engineered features:")
    for feature in new_features:
        if feature in df.columns:
            print(f"  {feature}: {df[feature].dtype}")
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    print(f"  Physical advantage range: {df['physical_advantage'].min():.2f} to {df['physical_advantage'].max():.2f}")
    print(f"  Win rate difference range: {df['win_rate_diff'].min():.2f} to {df['win_rate_diff'].max():.2f}")
    print(f"  Experience difference range: {df['experience_diff'].min():.0f} to {df['experience_diff'].max():.0f}")
    
    # Show target distribution
    print(f"\nTarget distribution (RESULT):")
    print(df['RESULT'].value_counts())
    
    print(f"\nFeature engineering demonstration completed successfully!")
    print(f"Dataset ready for modeling with {len(df.columns)} features.")

if __name__ == "__main__":
    main()
