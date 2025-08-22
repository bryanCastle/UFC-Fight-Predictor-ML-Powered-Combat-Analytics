"""
UFC ELO Rankings Script
Generates rankings of fighters with the highest ELO ratings
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def calculate_fighter_elo_ratings():
    """
    Calculate individual fighter ELO ratings from the ELO dataset
    """
    print("🏆 Calculating UFC Fighter ELO Rankings...")
    print("=" * 60)
    
    # Load the ELO dataset
    print("Loading ELO dataset...")
    df = pd.read_csv('dataUFC/ufc_elo_dataset.csv')
    print(f"✓ Loaded {len(df)} fights")
    
    # Initialize fighter ELO tracking
    fighter_elos = defaultdict(lambda: 1500)  # Default ELO of 1500
    fighter_weightclass_elos = defaultdict(lambda: defaultdict(lambda: 1500))
    fighter_names = {}
    fighter_weightclasses = {}
    fighter_fights = defaultdict(int)
    
    # Process each fight to build ELO ratings
    print("Processing fights to calculate ELO ratings...")
    
    for idx, row in df.iterrows():
        red_id = row['RedFighterID']
        blue_id = row['BlueFighterID']
        red_name = row['RedFighter']
        blue_name = row['BlueFighter']
        weight_class = row['WeightClass']
        result = row['RESULT']  # 1 = Red wins, 0 = Blue wins
        
        # Store fighter names
        fighter_names[red_id] = red_name
        fighter_names[blue_id] = blue_name
        fighter_weightclasses[red_id] = weight_class
        fighter_weightclasses[blue_id] = weight_class
        
        # Get current ELO ratings
        red_elo = fighter_elos[red_id]
        blue_elo = fighter_elos[blue_id]
        red_wc_elo = fighter_weightclass_elos[weight_class][red_id]
        blue_wc_elo = fighter_weightclass_elos[weight_class][blue_id]
        
        # Calculate expected probabilities
        k = 24  # Same K-factor as tennis model
        exp_red = 1/(1+(10**((blue_elo-red_elo)/400)))
        exp_blue = 1/(1+(10**((red_elo-blue_elo)/400)))
        exp_red_wc = 1/(1+(10**((blue_wc_elo-red_wc_elo)/400)))
        exp_blue_wc = 1/(1+(10**((red_wc_elo-blue_wc_elo)/400)))
        
        # Update ELO ratings
        if result == 1:  # Red wins
            fighter_elos[red_id] = red_elo + k*(1-exp_red)
            fighter_elos[blue_id] = blue_elo + k*(0-exp_blue)
            fighter_weightclass_elos[weight_class][red_id] = red_wc_elo + k*(1-exp_red_wc)
            fighter_weightclass_elos[weight_class][blue_id] = blue_wc_elo + k*(0-exp_blue_wc)
        else:  # Blue wins
            fighter_elos[red_id] = red_elo + k*(0-exp_red)
            fighter_elos[blue_id] = blue_elo + k*(1-exp_blue)
            fighter_weightclass_elos[weight_class][red_id] = red_wc_elo + k*(0-exp_red_wc)
            fighter_weightclass_elos[weight_class][blue_id] = blue_wc_elo + k*(1-exp_blue_wc)
        
        # Count fights
        fighter_fights[red_id] += 1
        fighter_fights[blue_id] += 1
    
    print(f"✓ Processed ELO ratings for {len(fighter_elos)} fighters")
    
    return fighter_elos, fighter_weightclass_elos, fighter_names, fighter_weightclasses, fighter_fights

def create_rankings(fighter_elos, fighter_weightclass_elos, fighter_names, fighter_weightclasses, fighter_fights):
    """
    Create comprehensive ELO rankings
    """
    print("\nCreating ELO Rankings...")
    
    # Create overall rankings
    overall_rankings = []
    for fighter_id, elo in fighter_elos.items():
        if fighter_id in fighter_names:
            overall_rankings.append({
                'Rank': 0,  # Will be set later
                'Fighter_ID': fighter_id,
                'Name': fighter_names[fighter_id],
                'Weight_Class': fighter_weightclasses.get(fighter_id, 'Unknown'),
                'Overall_ELO': elo,
                'WeightClass_ELO': fighter_weightclass_elos[fighter_weightclasses.get(fighter_id, 'Unknown')].get(fighter_id, 1500),
                'Total_Fights': fighter_fights[fighter_id]
            })
    
    # Sort by overall ELO
    overall_rankings.sort(key=lambda x: x['Overall_ELO'], reverse=True)
    
    # Add rankings
    for i, fighter in enumerate(overall_rankings):
        fighter['Rank'] = i + 1
    
    return overall_rankings

def display_rankings(rankings, top_n=50):
    """
    Display the rankings in a formatted way
    """
    print(f"\n🏆 TOP {top_n} UFC FIGHTERS BY ELO RATING")
    print("=" * 80)
    print(f"{'Rank':<4} {'Fighter Name':<25} {'Weight Class':<20} {'Overall ELO':<12} {'WC ELO':<10} {'Fights':<6}")
    print("-" * 80)
    
    for fighter in rankings[:top_n]:
        print(f"{fighter['Rank']:<4} {fighter['Name']:<25} {fighter['Weight_Class']:<20} "
              f"{fighter['Overall_ELO']:<12.1f} {fighter['WeightClass_ELO']:<10.1f} {fighter['Total_Fights']:<6}")
    
    print("-" * 80)

def create_weightclass_rankings(rankings):
    """
    Create rankings by weight class
    """
    print(f"\n⚖️ TOP FIGHTERS BY WEIGHT CLASS")
    print("=" * 80)
    
    # Group by weight class
    weightclass_fighters = defaultdict(list)
    for fighter in rankings:
        weightclass_fighters[fighter['Weight_Class']].append(fighter)
    
    # Display top 10 for each weight class
    for weight_class in sorted(weightclass_fighters.keys()):
        fighters = weightclass_fighters[weight_class]
        fighters.sort(key=lambda x: x['Overall_ELO'], reverse=True)
        
        if len(fighters) > 0:
            print(f"\n{weight_class.upper()} - Top 10")
            print("-" * 60)
            print(f"{'Rank':<4} {'Fighter Name':<25} {'Overall ELO':<12} {'WC ELO':<10} {'Fights':<6}")
            print("-" * 60)
            
            for i, fighter in enumerate(fighters[:10]):
                print(f"{i+1:<4} {fighter['Name']:<25} {fighter['Overall_ELO']:<12.1f} "
                      f"{fighter['WeightClass_ELO']:<10.1f} {fighter['Total_Fights']:<6}")

def save_rankings_to_csv(rankings):
    """
    Save rankings to CSV file
    """
    print(f"\nSaving rankings to CSV...")
    
    # Create DataFrame
    df_rankings = pd.DataFrame(rankings)
    
    # Reorder columns
    df_rankings = df_rankings[['Rank', 'Fighter_ID', 'Name', 'Weight_Class', 'Overall_ELO', 'WeightClass_ELO', 'Total_Fights']]
    
    # Save to file
    output_file = 'dataUFC/ufc_elo_rankings.csv'
    df_rankings.to_csv(output_file, index=False)
    print(f"✓ Saved rankings to {output_file}")
    
    return output_file

def analyze_rankings(rankings):
    """
    Analyze the rankings for insights
    """
    print(f"\nELO Rankings Analysis")
    print("=" * 50)
    
    # Basic statistics
    elos = [f['Overall_ELO'] for f in rankings]
    print(f"Total fighters ranked: {len(rankings)}")
    print(f"Average ELO: {np.mean(elos):.1f}")
    print(f"Highest ELO: {max(elos):.1f}")
    print(f"Lowest ELO: {min(elos):.1f}")
    print(f"ELO Standard Deviation: {np.std(elos):.1f}")
    
    # Weight class distribution
    weightclass_counts = defaultdict(int)
    for fighter in rankings:
        weightclass_counts[fighter['Weight_Class']] += 1
    
    print(f"\n🏆 Top 10 Weight Classes by Fighter Count:")
    sorted_wc = sorted(weightclass_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (wc, count) in enumerate(sorted_wc[:10]):
        print(f"{i+1:2d}. {wc:<20} {count:3d} fighters")
    
    # Experience analysis
    fights = [f['Total_Fights'] for f in rankings]
    print(f"\nFight Experience Analysis:")
    print(f"Average fights per fighter: {np.mean(fights):.1f}")
    print(f"Most experienced fighter: {max(fights)} fights")
    print(f"Least experienced fighter: {min(fights)} fights")
    
    # Top fighters by experience
    print(f"\nTop 10 Most Experienced Fighters:")
    experienced_fighters = sorted(rankings, key=lambda x: x['Total_Fights'], reverse=True)[:10]
    for i, fighter in enumerate(experienced_fighters):
        print(f"{i+1:2d}. {fighter['Name']:<25} {fighter['Total_Fights']:3d} fights (ELO: {fighter['Overall_ELO']:.1f})")

def main():
    """
    Main function to generate and display ELO rankings
    """
    print("UFC ELO Rankings Generator")
    print("=" * 60)
    
    # Calculate ELO ratings
    fighter_elos, fighter_weightclass_elos, fighter_names, fighter_weightclasses, fighter_fights = calculate_fighter_elo_ratings()
    
    # Create rankings
    rankings = create_rankings(fighter_elos, fighter_weightclass_elos, fighter_names, fighter_weightclasses, fighter_fights)
    
    # Display overall rankings
    display_rankings(rankings, top_n=50)
    
    # Display weight class rankings
    create_weightclass_rankings(rankings)
    
    # Analyze rankings
    analyze_rankings(rankings)
    
    # Save to CSV
    output_file = save_rankings_to_csv(rankings)
    
    print(f"\nELO Rankings Generation Complete!")
    print(f"Rankings saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
