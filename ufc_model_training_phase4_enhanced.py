#!/usr/bin/env python3
"""
UFC Model Training - Phase 4 (Enhanced)
Phase 4: Model Training with Combined ELO + Engineered Features
Combines ELO features with original engineered features for optimal performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive mode
plt.ioff()

def load_and_combine_datasets():
    """Load and combine ELO and engineered datasets"""
    print("Loading and combining datasets...")
    
    # Load ELO dataset
    try:
        elo_df = pd.read_csv("dataUFC/ufc_final_with_elo.csv")
        print(f"✓ Loaded ELO dataset: {elo_df.shape}")
    except FileNotFoundError:
        print("❌ ELO dataset not found!")
        return None
    
    # Load engineered dataset
    try:
        eng_df = pd.read_csv("dataUFC/ufc_engineered.csv")
        print(f"✓ Loaded engineered dataset: {eng_df.shape}")
    except FileNotFoundError:
        print("❌ Engineered dataset not found!")
        return None
    
    # Create a combined dataset
    print("Combining datasets...")
    
    # Use engineered dataset as base (has better target distribution)
    combined_df = eng_df.copy()
    
    # Add ELO features
    elo_features = [col for col in elo_df.columns if 'ELO' in col]
    for feature in elo_features:
        if feature not in combined_df.columns:
            # Try to match by fighter IDs and date
            combined_df = combined_df.merge(
                elo_df[['RedFighterID', 'BlueFighterID', 'Date', feature]], 
                on=['RedFighterID', 'BlueFighterID', 'Date'], 
                how='left'
            )
    
    print(f"✓ Combined dataset: {combined_df.shape}")
    
    # Check ELO feature integration
    integrated_elo_features = [col for col in combined_df.columns if 'ELO' in col]
    print(f"✓ Integrated ELO features: {len(integrated_elo_features)}")
    
    return combined_df

def prepare_data(df):
    """Prepare data for training"""
    print("Preparing data for training...")
    
    # Select only numeric columns for training
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove identifier columns and target from features
    exclude_cols = ['RedFighterID', 'BlueFighterID', 'RESULT']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"✓ Selected {len(feature_cols)} features for training")
    
    # Check for missing values
    missing_counts = df[feature_cols].isnull().sum()
    features_with_missing = missing_counts[missing_counts > 0]
    
    if len(features_with_missing) > 0:
        print(f"⚠️ Features with missing values: {len(features_with_missing)}")
        # Fill missing values with 0 for ELO features, median for others
        for feature in feature_cols:
            if 'ELO' in feature:
                df[feature] = df[feature].fillna(0)
            else:
                df[feature] = df[feature].fillna(df[feature].median())
        print("✓ Filled missing values")
    
    # Prepare X and y
    X = df[feature_cols]
    y = df['RESULT']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train all models"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    models = {}
    results = {}
    
    # Decision Tree
    print("\n🌳 Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    models['Decision Tree'] = dt
    results['Decision Tree'] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'predictions': y_pred, 'probabilities': y_pred_proba}
    
    print(f"✓ Decision Tree Accuracy: {accuracy:.4f}")
    print(f"✓ Decision Tree ROC AUC: {roc_auc:.4f}")
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'predictions': y_pred, 'probabilities': y_pred_proba}
    
    print(f"✓ Random Forest Accuracy: {accuracy:.4f}")
    print(f"✓ Random Forest ROC AUC: {roc_auc:.4f}")
    
    # Optimized Random Forest
    print("\nOptimizing Random Forest...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [7, 10],
        'min_samples_split': [2, 5]
    }
    
    rf_opt = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_opt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    models['Random Forest Optimized'] = best_rf
    results['Random Forest Optimized'] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'predictions': y_pred, 'probabilities': y_pred_proba}
    
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Optimized Random Forest Accuracy: {accuracy:.4f}")
    print(f"✓ Optimized Random Forest ROC AUC: {roc_auc:.4f}")
    
    # XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {'accuracy': accuracy, 'roc_auc': roc_auc, 'predictions': y_pred, 'probabilities': y_pred_proba}
    
    print(f"✓ XGBoost Accuracy: {accuracy:.4f}")
    print(f"✓ XGBoost ROC AUC: {roc_auc:.4f}")
    
    return models, results

def analyze_feature_importance(models, feature_cols):
    """Analyze feature importance for all models"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    for model_name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n{model_name} Feature Importance:")
            
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Show top 15 features
            print(f"\nTop 15 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
            
            # Categorize features
            elo_features = [f for f in feature_importance['feature'] if 'ELO' in f]
            top_elo_features = [f for f in elo_features if f in feature_importance.head(20)['feature'].values]
            
            print(f"\nELO features in top 20: {len(top_elo_features)}")
            for feature in top_elo_features:
                importance = feature_importance[feature_importance['feature'] == feature]['importance'].iloc[0]
                print(f"  - {feature}: {importance:.4f}")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top 15 Feature Importance (Phase 4 Enhanced)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'images/{model_name.lower().replace(" ", "_")}_feature_importance_phase4_enhanced.png', dpi=300, bbox_inches='tight')
            plt.close()

def evaluate_performance(results, y_test):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n📈 {model_name} Performance:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"ROC AUC: {result['roc_auc']:.4f}")
        
        # Classification Report
        print(f"\nClassification Report:")
        print(classification_report(y_test, result['predictions']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, result['predictions'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fighter2 Wins', 'Fighter1 Wins'],
                    yticklabels=['Fighter2 Wins', 'Fighter1 Wins'])
        plt.title(f'{model_name} - Confusion Matrix (Phase 4 Enhanced)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'images/{model_name.lower().replace(" ", "_")}_confusion_matrix_phase4_enhanced.png', dpi=300, bbox_inches='tight')
        plt.close()

def compare_with_previous_phases(results):
    """Compare with previous phases"""
    print("\n" + "="*60)
    print("COMPARISON WITH PREVIOUS PHASES")
    print("="*60)
    
    # Previous phase results
    previous_results = {
        'Decision Tree': 0.8408,
        'Random Forest': 0.9891,
        'XGBoost': 0.9952
    }
    
    print(f"\nPerformance Comparison:")
    print("=" * 70)
    print(f"{'Model':<25} {'Previous':<12} {'Phase 4':<12} {'Change':<12} {'Status'}")
    print("-" * 70)
    
    for model_name, result in results.items():
        current_acc = result['accuracy']
        
        if model_name in previous_results:
            prev_acc = previous_results[model_name]
            change = current_acc - prev_acc
            change_str = f"{change:+.4f}"
            
            if change >= 0:
                status = "✅ Improved"
            else:
                status = "⚠️ Decreased"
                
            print(f"{model_name:<25} {prev_acc:<12.4f} {current_acc:<12.4f} {change_str:<12} {status}")
        else:
            print(f"{model_name:<25} {'N/A':<12} {current_acc:<12.4f} {'N/A':<12} {'🆕 New'}")
    
    print("=" * 70)

def save_models(models):
    """Save trained models"""
    print("\nSaving models...")
    
    for model_name, model in models.items():
        filename = f"models/{model_name.lower().replace(' ', '_')}_phase4_enhanced.joblib"
        joblib.dump(model, filename)
        print(f"✓ Saved {model_name}: {filename}")

def main():
    """Main training function"""
    print("=== UFC Model Training - Phase 4 (Enhanced) ===")
    print("Combining ELO features with engineered features for optimal performance\n")
    
    # Load and combine datasets
    combined_df = load_and_combine_datasets()
    if combined_df is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(combined_df)
    
    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test, feature_cols)
    
    # Analyze feature importance
    analyze_feature_importance(models, feature_cols)
    
    # Evaluate performance
    evaluate_performance(results, y_test)
    
    # Compare with previous phases
    compare_with_previous_phases(results)
    
    # Save models
    save_models(models)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    # Final summary
    print(f"\nPhase 4 Enhanced Training Complete!")
    print(f"Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print(f"Models saved to models/ directory")
    print(f"Performance plots saved to images/ directory")
    
    # Create completion marker
    with open("models/PHASE4_ENHANCED_TRAINING_DONE.txt", "w") as f:
        f.write(f"Phase 4 Enhanced Training Complete\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Best Accuracy: {best_accuracy:.4f}\n")
        f.write(f"Features Used: {len(feature_cols)}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"ELO Features Integrated: {len([f for f in feature_cols if 'ELO' in f])}\n")
    
    print(f"✅ Training completion marker created")

if __name__ == "__main__":
    main()
