"""
Show Top UFC Fighters by ELO Rating
Simple script to display the highest-rated fighters
"""

import pandas as pd

def show_top_fighters(top_n=20):
    """
    Display the top fighters by ELO rating
    """
    try:
        # Load the rankings
        df = pd.read_csv('dataUFC/ufc_elo_rankings.csv')
        
        print(f"🏆 TOP {top_n} UFC FIGHTERS BY ELO RATING")
        print("=" * 70)
        print(f"{'Rank':<4} {'Fighter Name':<25} {'Weight Class':<18} {'ELO':<8} {'Fights':<6}")
        print("-" * 70)
        
        for idx, row in df.head(top_n).iterrows():
            print(f"{row['Rank']:<4} {row['Name']:<25} {row['Weight_Class']:<18} {row['Overall_ELO']:<8.1f} {row['Total_Fights']:<6}")
        
        print("-" * 70)
        
        # Show some interesting stats
        print(f"\n📊 Quick Stats:")
        print(f"Total fighters ranked: {len(df)}")
        print(f"Highest ELO: {df['Overall_ELO'].max():.1f} ({df.loc[df['Overall_ELO'].idxmax(), 'Name']})")
        print(f"Average ELO: {df['Overall_ELO'].mean():.1f}")
        
        # Show top fighter by weight class
        print(f"\n Top Fighter by Weight Class:")
        top_by_wc = df.loc[df.groupby('Weight_Class')['Overall_ELO'].idxmax()]
        for _, fighter in top_by_wc.head(10).iterrows():
            print(f"{fighter['Weight_Class']:<20}: {fighter['Name']:<20} (ELO: {fighter['Overall_ELO']:.1f})")
            
    except FileNotFoundError:
        print("❌ Rankings file not found! Please run ufc_elo_rankings.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_top_fighters(20)
