"""
UFC ELO Dataset Creator
Creates UFC dataset with ELO ratings by processing cleaned UFC data
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.ufc_elo_utils import getUFCStats, updateUFCStats, createUFCStats
import warnings
warnings.filterwarnings('ignore')

def create_ufc_elo_dataset():
    """
    Create UFC dataset with ELO ratings
    
    Returns:
        pd.DataFrame: Final dataset with ELO features
    """
    print("Starting UFC ELO Dataset Creation...")
    print("=" * 50)
    
    # Load cleaned UFC data (equivalent to 0cleanDataset.csv in tennis)
    print("Loading cleaned UFC data...")
    try:
        ufc_data = pd.read_csv("dataUFC/ufc_cleaned.csv")
        print(f"✓ Loaded {len(ufc_data)} fights from ufc_cleaned.csv")
    except FileNotFoundError:
        print("❌ Error: ufc_cleaned.csv not found!")
        print("Please run ufc_data_cleaning.py first to create the cleaned dataset.")
        return None
    
    # Sort by date to ensure chronological order for ELO calculations
    if 'Date' in ufc_data.columns:
        ufc_data['Date'] = pd.to_datetime(ufc_data['Date'], errors='coerce')
        ufc_data = ufc_data.sort_values('Date').reset_index(drop=True)
        print("✓ Sorted fights chronologically by date")
    
    # Initialize stats structure
    print("🏗️ Initializing ELO stats structure...")
    prev_stats = createUFCStats()
    
    final_dataset = []
    
    print("🔄 Processing fights and calculating ELO ratings...")
    # Iterate through each row in UFC data - EXACT SAME as tennis
    for index, row in tqdm(ufc_data.iterrows(), total=len(ufc_data), desc="Processing fights"):
        try:
            # Convert Winner to RESULT format (1 = Red wins, 0 = Blue wins)
            result = 1 if row["Winner"] == "Red" else 0
            
            # Create fighter dictionaries - EXACT SAME structure as tennis
            fighter1 = {
                "ID": row["RedFighterID"],
                "AGE": row["RedAge"],
                "HEIGHT": row["RedHeightCms"],
                "REACH": row["RedReachCms"],
                "WEIGHT": row["RedWeightLbs"],
            }

            fighter2 = {
                "ID": row["BlueFighterID"],
                "AGE": row["BlueAge"],
                "HEIGHT": row["BlueHeightCms"],
                "REACH": row["BlueReachCms"],
                "WEIGHT": row["BlueWeightLbs"],
            }

            fight = {
                "WeightClass": row["WeightClass"],
                "NumberOfRounds": row["NumberOfRounds"],
                "Gender": row["Gender"],
            }

            ########## GET STATS ##########
            # Call getUFCStats function - EXACT SAME as tennis
            output = getUFCStats(fighter1, fighter2, fight, prev_stats)

            # Append sorted stats to final dataset
            match_data = dict(sorted(output.items()))
            match_data["RESULT"] = result
            
            # Add original fight identifiers for reference
            match_data["RedFighterID"] = row["RedFighterID"]
            match_data["BlueFighterID"] = row["BlueFighterID"]
            match_data["RedFighter"] = row["RedFighter"]
            match_data["BlueFighter"] = row["BlueFighter"]
            match_data["Date"] = row.get("Date", "")
            match_data["WeightClass"] = row["WeightClass"]
            
            final_dataset.append(match_data)

            ########## UPDATE STATS ##########
            # Create a modified row with RESULT column for updateUFCStats
            row_with_result = row.copy()
            row_with_result["RESULT"] = result
            prev_stats = updateUFCStats(row_with_result, prev_stats)
            
        except Exception as e:
            print(f"⚠️ Error processing fight {index}: {e}")
            continue

    # Convert final dataset to DataFrame
    print("Converting to DataFrame...")
    final_dataset = pd.DataFrame(final_dataset)
    
    print(f"✓ Created dataset with {len(final_dataset)} fights and {len(final_dataset.columns)} features")
    
    # Display feature information
    print("\n Dataset Information:")
    print(f"Shape: {final_dataset.shape}")
    print(f"Features: {len(final_dataset.columns)}")
    
    # Show ELO-related features
    elo_features = [col for col in final_dataset.columns if 'ELO' in col]
    print(f"\n ELO Features ({len(elo_features)}):")
    for feature in elo_features:
        print(f"  - {feature}")
    
    # Show win/loss features
    win_features = [col for col in final_dataset.columns if 'WIN_LAST' in col]
    print(f"\n🏆 Win/Loss Features ({len(win_features)}):")
    for feature in win_features:
        print(f"  - {feature}")
    
    # Show gradient features
    grad_features = [col for col in final_dataset.columns if 'ELO_GRAD' in col]
    print(f"\nELO Gradient Features ({len(grad_features)}):")
    for feature in grad_features:
        print(f"  - {feature}")
    
    # Show UFC-specific features
    ufc_features = [col for col in final_dataset.columns if any(x in col for x in ['STRIKE', 'TD_ACC', 'KO_RATE', 'SUB_RATE'])]
    print(f"\nUFC-Specific Features ({len(ufc_features)}):")
    for feature in ufc_features:
        print(f"  - {feature}")
    
    # Save to file
    print("\nSaving ELO dataset...")
    output_path = "dataUFC/ufc_elo_dataset.csv"
    final_dataset.to_csv(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    # Display sample of ELO ratings
    print("\nSample ELO Ratings:")
    if len(prev_stats["elo_fighters"]) > 0:
        # Get top 10 fighters by ELO rating
        top_fighters = sorted(prev_stats["elo_fighters"].items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top 10 Fighters by ELO Rating:")
        for fighter_id, elo_rating in top_fighters:
            print(f"  Fighter {fighter_id}: {elo_rating:.1f}")
    
    # Display weight class ELO distribution
    print("\nWeight Class ELO Distribution:")
    for weight_class in prev_stats["elo_weightclass_fighters"]:
        if len(prev_stats["elo_weightclass_fighters"][weight_class]) > 0:
            ratings = list(prev_stats["elo_weightclass_fighters"][weight_class].values())
            avg_rating = np.mean(ratings)
            print(f"  {weight_class}: {len(ratings)} fighters, avg ELO: {avg_rating:.1f}")
    
    print("\nUFC ELO Dataset Creation Complete!")
    print("=" * 50)
    
    return final_dataset


def validate_elo_dataset(dataset):
    """
    Validate the created ELO dataset
    
    Args:
        dataset: The created ELO dataset
        
    Returns:
        bool: True if validation passes
    """
    print("\nValidating ELO Dataset...")
    
    # Check required features
    required_features = [
        "ELO_DIFF", "ELO_WEIGHTCLASS_DIFF", "N_FIGHTS_DIFF",
        "H2H_DIFF", "H2H_WEIGHTCLASS_DIFF"
    ]
    
    missing_features = [f for f in required_features if f not in dataset.columns]
    if missing_features:
        print(f"❌ Missing required features: {missing_features}")
        return False
    
    # Check for expected K-value features
    expected_k_features = []
    for k in [3, 5, 10, 25, 50, 100, 200]:
        expected_k_features.extend([
            f"WIN_LAST_{k}_DIFF",
            f"ELO_GRAD_LAST_{k}_DIFF",
            f"STRIKE_ACC_LAST_{k}_DIFF",
            f"TD_ACC_LAST_{k}_DIFF",
            f"KO_RATE_LAST_{k}_DIFF",
            f"SUB_RATE_LAST_{k}_DIFF"
        ])
    
    missing_k_features = [f for f in expected_k_features if f not in dataset.columns]
    if missing_k_features:
        print(f"❌ Missing K-value features: {missing_k_features[:5]}...")
        return False
    
    # Check for reasonable ELO values
    if "ELO_DIFF" in dataset.columns:
        elo_diff_stats = dataset["ELO_DIFF"].describe()
        print(f"✓ ELO_DIFF statistics:")
        print(f"  - Min: {elo_diff_stats['min']:.1f}")
        print(f"  - Max: {elo_diff_stats['max']:.1f}")
        print(f"  - Mean: {elo_diff_stats['mean']:.1f}")
        print(f"  - Std: {elo_diff_stats['std']:.1f}")
    
    # Check for no infinite or NaN values
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    inf_count = dataset[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
    nan_count = dataset[numeric_cols].isna().sum().sum()
    
    if inf_count > 0:
        print(f"⚠️ Found {inf_count} infinite values")
    if nan_count > 0:
        print(f"⚠️ Found {nan_count} NaN values")
    
    print("✓ ELO dataset validation complete!")
    return True


if __name__ == "__main__":
    # Create the ELO dataset
    elo_dataset = create_ufc_elo_dataset()
    
    if elo_dataset is not None:
        # Validate the dataset
        validation_passed = validate_elo_dataset(elo_dataset)
        
        if validation_passed:
            print("\n✅ UFC ELO Dataset Creation and Validation Successful!")
            print("Ready to proceed to Phase 3: Feature Engineering Integration")
        else:
            print("\n❌ Validation failed. Please check the dataset.")
    else:
        print("\n❌ Dataset creation failed.")
