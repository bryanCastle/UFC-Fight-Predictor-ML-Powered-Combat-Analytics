#!/usr/bin/env python3
"""
UFC Prediction System
Phase 4: Prediction System Implementation
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UFCPredictionSystem:
    """UFC Fight Prediction System"""
    
    def __init__(self, model_path="models/ufc_xgboost.joblib"):
        """Initialize the prediction system"""
        print("Loading UFC Prediction System...")
        
        # Load the best model (XGBoost)
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        
        # Load feature engineering functions
        self.feature_columns = None
        self.load_feature_columns()
        
    def load_feature_columns(self):
        """Load the feature columns used in training"""
        # Load a sample of the engineered data to get feature names
        try:
            df = pd.read_csv("dataUFC/ufc_engineered.csv")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numeric_columns if col != 'RESULT']
            print(f"Loaded {len(self.feature_columns)} feature columns")
        except Exception as e:
            print(f"Warning: Could not load feature columns: {e}")
            # Fallback to common UFC features
            self.feature_columns = [
                'RedOdds', 'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue',
                'experience_diff', 'win_rate_diff', 'physical_advantage',
                'height_diff', 'reach_diff', 'weight_diff', 'age_diff',
                'ko_rate_diff', 'sub_rate_diff', 'weight_class_encoded',
                'gender_encoded', 'title_bout_encoded', 'better_rank_encoded'
            ]
    
    def prepare_fight_data(self, fighter1_data, fighter2_data, fight_context=None):
        """Prepare fight data for prediction"""
        print("Preparing fight data for prediction...")
        
        # Create feature dictionary
        features = {}
        
        # Physical advantages
        features['height_diff'] = fighter1_data.get('height_cm', 0) - fighter2_data.get('height_cm', 0)
        features['reach_diff'] = fighter1_data.get('reach_cm', 0) - fighter2_data.get('reach_cm', 0)
        features['weight_diff'] = fighter1_data.get('weight_lbs', 0) - fighter2_data.get('weight_lbs', 0)
        features['age_diff'] = fighter1_data.get('age', 0) - fighter2_data.get('age', 0)
        
        # Experience metrics
        fighter1_total_fights = fighter1_data.get('wins', 0) + fighter1_data.get('losses', 0) + fighter1_data.get('draws', 0)
        fighter2_total_fights = fighter2_data.get('wins', 0) + fighter2_data.get('losses', 0) + fighter2_data.get('draws', 0)
        features['experience_diff'] = fighter1_total_fights - fighter2_total_fights
        
        # Win rates
        fighter1_win_rate = fighter1_data.get('wins', 0) / (fighter1_total_fights + 1)
        fighter2_win_rate = fighter2_data.get('wins', 0) / (fighter2_total_fights + 1)
        features['win_rate_diff'] = fighter1_win_rate - fighter2_win_rate
        
        # Win method rates
        fighter1_ko_rate = fighter1_data.get('wins_by_ko', 0) / (fighter1_data.get('wins', 0) + 1)
        fighter2_ko_rate = fighter2_data.get('wins_by_ko', 0) / (fighter2_data.get('wins', 0) + 1)
        features['ko_rate_diff'] = fighter1_ko_rate - fighter2_ko_rate
        
        fighter1_sub_rate = fighter1_data.get('wins_by_submission', 0) / (fighter1_data.get('wins', 0) + 1)
        fighter2_sub_rate = fighter2_data.get('wins_by_submission', 0) / (fighter2_data.get('wins', 0) + 1)
        features['sub_rate_diff'] = fighter1_sub_rate - fighter2_sub_rate
        
        # Physical advantage composite
        features['physical_advantage'] = (
            (features['height_diff'] * 0.3) + 
            (features['reach_diff'] * 0.4) + 
            (features['weight_diff'] * 0.2) + 
            (features['age_diff'] * 0.1)
        ) / 4
        
        # Fight context
        if fight_context:
            features['weight_class_encoded'] = fight_context.get('weight_class_encoded', 5)  # Default to Welterweight
            features['gender_encoded'] = fight_context.get('gender_encoded', 1)  # Default to Male
            features['title_bout_encoded'] = fight_context.get('title_bout_encoded', 0)  # Default to non-title
            features['better_rank_encoded'] = fight_context.get('better_rank_encoded', 0)  # Default to neither
        else:
            features['weight_class_encoded'] = 5
            features['gender_encoded'] = 1
            features['title_bout_encoded'] = 0
            features['better_rank_encoded'] = 0
        
        # Betting odds (if available)
        features['RedOdds'] = fighter1_data.get('odds', 100)
        features['BlueOdds'] = fighter2_data.get('odds', 100)
        features['RedExpectedValue'] = fighter1_data.get('expected_value', 0)
        features['BlueExpectedValue'] = fighter2_data.get('expected_value', 0)
        
        # Calculate odds differences
        features['odds_diff'] = features['RedOdds'] - features['BlueOdds']
        features['ev_diff'] = features['RedExpectedValue'] - features['BlueExpectedValue']
        
        # Method-specific odds (if available)
        features['RedDecOdds'] = fighter1_data.get('decision_odds', 100)
        features['BlueDecOdds'] = fighter2_data.get('decision_odds', 100)
        features['RSubOdds'] = fighter1_data.get('submission_odds', 100)
        features['BSubOdds'] = fighter2_data.get('submission_odds', 100)
        features['RKOOdds'] = fighter1_data.get('ko_odds', 100)
        features['BKOOdds'] = fighter2_data.get('ko_odds', 100)
        
        # Calculate method-specific differences
        features['dec_odds_diff'] = features['RedDecOdds'] - features['BlueDecOdds']
        features['sub_odds_diff'] = features['RSubOdds'] - features['BSubOdds']
        features['ko_odds_diff'] = features['RKOOdds'] - features['BKOOdds']
        
        # Additional features (set to defaults if not available)
        features['NumberOfRounds'] = fight_context.get('number_of_rounds', 3) if fight_context else 3
        features['TotalRoundDif'] = 0  # Default
        features['KODif'] = features['ko_rate_diff']  # Use KO rate difference
        features['WinDif'] = features['win_rate_diff']  # Use win rate difference
        features['AvgSubAttDif'] = features['sub_rate_diff']  # Use submission rate difference
        
        # Performance metrics (if available)
        features['RedAvgSigStrPct'] = fighter1_data.get('strike_accuracy', 50)
        features['BlueAvgSigStrPct'] = fighter2_data.get('strike_accuracy', 50)
        features['RedAvgTDPct'] = fighter1_data.get('takedown_accuracy', 50)
        features['BlueAvgTDPct'] = fighter2_data.get('takedown_accuracy', 50)
        features['RedAvgSigStrLanded'] = fighter1_data.get('strikes_landed', 100)
        features['BlueAvgSigStrLanded'] = fighter2_data.get('strikes_landed', 100)
        features['RedAvgSubAtt'] = fighter1_data.get('submission_attempts', 1)
        features['BlueAvgSubAtt'] = fighter2_data.get('submission_attempts', 1)
        
        # Calculate performance differences
        features['strike_acc_diff'] = features['RedAvgSigStrPct'] - features['BlueAvgSigStrPct']
        features['td_acc_diff'] = features['RedAvgTDPct'] - features['BlueAvgTDPct']
        features['strike_landed_diff'] = features['RedAvgSigStrLanded'] - features['BlueAvgSigStrLanded']
        features['sub_att_diff'] = features['RedAvgSubAtt'] - features['BlueAvgSubAtt']
        
        # Career statistics
        features['RedWins'] = fighter1_data.get('wins', 0)
        features['BlueWins'] = fighter2_data.get('wins', 0)
        features['RedLosses'] = fighter1_data.get('losses', 0)
        features['BlueLosses'] = fighter2_data.get('losses', 0)
        features['RedDraws'] = fighter1_data.get('draws', 0)
        features['BlueDraws'] = fighter2_data.get('draws', 0)
        features['RedWinsByKO'] = fighter1_data.get('wins_by_ko', 0)
        features['BlueWinsByKO'] = fighter2_data.get('wins_by_ko', 0)
        features['RedWinsBySubmission'] = fighter1_data.get('wins_by_submission', 0)
        features['BlueWinsBySubmission'] = fighter2_data.get('wins_by_submission', 0)
        
        # Fill missing features with zeros
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0
        
        return features
    
    def predict_fight(self, fighter1_data, fighter2_data, fight_context=None):
        """Predict the outcome of a UFC fight"""
        print(f"Predicting fight: {fighter1_data.get('name', 'Fighter 1')} vs {fighter2_data.get('name', 'Fighter 2')}")
        
        # Prepare fight data
        features = self.prepare_fight_data(fighter1_data, fighter2_data, fight_context)
        
        # Create feature vector
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(feature_vector)[0]
        prediction_proba = self.model.predict_proba(feature_vector)[0]
        
        # Convert prediction to readable format
        if prediction == 1:
            winner = fighter1_data.get('name', 'Fighter 1')
            winner_prob = prediction_proba[1]
            loser_prob = prediction_proba[0]
        else:
            winner = fighter2_data.get('name', 'Fighter 2')
            winner_prob = prediction_proba[0]
            loser_prob = prediction_proba[1]
        
        # Create result dictionary
        result = {
            'winner': winner,
            'winner_probability': winner_prob,
            'loser_probability': loser_prob,
            'confidence': max(winner_prob, loser_prob),
            'prediction': prediction,
            'features_used': len(self.feature_columns)
        }
        
        return result
    
    def predict_multiple_fights(self, fights_data):
        """Predict multiple fights"""
        print(f"Predicting {len(fights_data)} fights...")
        
        predictions = []
        for i, fight in enumerate(fights_data):
            print(f"\nFight {i+1}: {fight['fighter1']['name']} vs {fight['fighter2']['name']}")
            
            result = self.predict_fight(
                fight['fighter1'], 
                fight['fighter2'], 
                fight.get('context')
            )
            
            predictions.append({
                'fight_number': i+1,
                'fighter1': fight['fighter1']['name'],
                'fighter2': fight['fighter2']['name'],
                'prediction': result
            })
        
        return predictions
    
    def analyze_prediction_confidence(self, prediction_result):
        """Analyze the confidence of a prediction"""
        confidence = prediction_result['confidence']
        
        if confidence >= 0.9:
            confidence_level = "Very High"
        elif confidence >= 0.8:
            confidence_level = "High"
        elif confidence >= 0.7:
            confidence_level = "Medium"
        elif confidence >= 0.6:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': confidence,
            'recommendation': f"Prediction confidence: {confidence_level} ({confidence:.1%})"
        }

def create_sample_fight_data():
    """Create sample fight data for testing"""
    return {
        'fighter1': {
            'name': 'Conor McGregor',
            'age': 35,
            'height_cm': 175,
            'reach_cm': 188,
            'weight_lbs': 155,
            'wins': 22,
            'losses': 6,
            'draws': 0,
            'wins_by_ko': 19,
            'wins_by_submission': 1,
            'odds': 150,
            'expected_value': 0.05,
            'decision_odds': 200,
            'submission_odds': 300,
            'ko_odds': 120,
            'strike_accuracy': 45,
            'takedown_accuracy': 30,
            'strikes_landed': 120,
            'submission_attempts': 2
        },
        'fighter2': {
            'name': 'Dustin Poirier',
            'age': 34,
            'height_cm': 175,
            'reach_cm': 183,
            'weight_lbs': 155,
            'wins': 29,
            'losses': 7,
            'draws': 0,
            'wins_by_ko': 15,
            'wins_by_submission': 7,
            'odds': 120,
            'expected_value': 0.08,
            'decision_odds': 180,
            'submission_odds': 250,
            'ko_odds': 140,
            'strike_accuracy': 50,
            'takedown_accuracy': 45,
            'strikes_landed': 140,
            'submission_attempts': 3
        },
        'context': {
            'weight_class_encoded': 4,  # Lightweight
            'gender_encoded': 1,  # Male
            'title_bout_encoded': 0,  # Non-title
            'better_rank_encoded': 0,  # Neither
            'number_of_rounds': 3
        }
    }

def main():
    """Main function to demonstrate the prediction system"""
    print("=== UFC Prediction System Demo ===\n")
    
    # Initialize prediction system
    predictor = UFCPredictionSystem()
    
    # Create sample fight data
    sample_fight = create_sample_fight_data()
    
    # Make prediction
    result = predictor.predict_fight(
        sample_fight['fighter1'],
        sample_fight['fighter2'],
        sample_fight['context']
    )
    
    # Display results
    print(f"\n=== Prediction Results ===")
    print(f"Winner: {result['winner']}")
    print(f"Winner Probability: {result['winner_probability']:.1%}")
    print(f"Loser Probability: {result['loser_probability']:.1%}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Features Used: {result['features_used']}")
    
    # Analyze confidence
    confidence_analysis = predictor.analyze_prediction_confidence(result)
    print(f"\nConfidence Analysis:")
    print(f"Level: {confidence_analysis['confidence_level']}")
    print(f"Score: {confidence_analysis['confidence_score']:.1%}")
    print(f"Recommendation: {confidence_analysis['recommendation']}")
    
    print(f"\nPrediction system demo completed successfully!")

if __name__ == "__main__":
    main()
