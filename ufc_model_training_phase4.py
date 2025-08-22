#!/usr/bin/env python3
"""
UFC Model Training - Phase 4
Phase 4: Model Training with ELO Features
Trains models on the ELO-enhanced dataset and compares with previous performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive mode
plt.ioff()

def load_elo_dataset():
    """Load the ELO-enhanced dataset"""
    print("Loading ELO-enhanced dataset...")
    try:
        df = pd.read_csv("dataUFC/ufc_final_with_elo.csv")
        print(f"✓ Loaded ELO dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ Error: ufc_final_with_elo.csv not found!")
        print("Please run Phase 3 first to create the ELO-enhanced dataset.")
        return None

def prepare_data(df):
    """Prepare data for training"""
    print("Preparing data for training...")
    
    # Select only numeric columns for training
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove identifier columns and target from features
    exclude_cols = ['RedFighterID', 'BlueFighterID', 'RESULT']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"✓ Selected {len(feature_cols)} features for training")
    
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

def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train Decision Tree model"""
    print("\n🌳 Training Decision Tree...")
    
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    
    # Predictions
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✓ Decision Tree Accuracy: {accuracy:.4f}")
    print(f"✓ Decision Tree ROC AUC: {roc_auc:.4f}")
    
    return dt, accuracy, roc_auc, y_pred, y_pred_proba

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model"""
    print("\n🌲 Training Random Forest...")
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✓ Random Forest Accuracy: {accuracy:.4f}")
    print(f"✓ Random Forest ROC AUC: {roc_auc:.4f}")
    
    return rf, accuracy, roc_auc, y_pred, y_pred_proba

def optimize_random_forest(X_train, X_test, y_train, y_test):
    """Optimize Random Forest with Grid Search"""
    print("\nOptimizing Random Forest with Grid Search...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 7, 10],
        'min_samples_split': [2, 5, 10]
    }
    
    # Grid search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Optimized Random Forest Accuracy: {accuracy:.4f}")
    print(f"✓ Optimized Random Forest ROC AUC: {roc_auc:.4f}")
    
    return best_rf, accuracy, roc_auc, y_pred, y_pred_proba

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model"""
    print("\nTraining XGBoost...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✓ XGBoost Accuracy: {accuracy:.4f}")
    print(f"✓ XGBoost ROC AUC: {roc_auc:.4f}")
    
    return xgb_model, accuracy, roc_auc, y_pred, y_pred_proba

def analyze_feature_importance(model, feature_cols, model_name):
    """Analyze feature importance"""
    print(f"\nAnalyzing {model_name} Feature Importance...")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Display top 15 features
    print(f"\nTop 15 Most Important Features ({model_name}):")
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'images/{model_name.lower().replace(" ", "_")}_feature_importance_phase4.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance

def evaluate_model_performance(y_test, y_pred, y_pred_proba, model_name):
    """Evaluate model performance with detailed metrics"""
    print(f"\n{model_name} Performance Evaluation:")
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fighter2 Wins', 'Fighter1 Wins'],
                yticklabels=['Fighter2 Wins', 'Fighter1 Wins'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'images/{model_name.lower().replace(" ", "_")}_confusion_matrix_phase4.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy, roc_auc

def save_models(models_dict):
    """Save trained models"""
    print("\nSaving models...")
    
    for model_name, model in models_dict.items():
        filename = f"models/{model_name.lower().replace(' ', '_')}_phase4.joblib"
        joblib.dump(model, filename)
        print(f"✓ Saved {model_name}: {filename}")

def compare_with_previous_models(results_dict):
    """Compare Phase 4 results with previous models"""
    print("\nComparing with Previous Models...")
    
    # Load previous results if available
    try:
        previous_results = {
            'Decision Tree': 0.8408,  # From previous phase
            'Random Forest': 0.9891,  # From previous phase
            'XGBoost': 0.9952        # From previous phase
        }
        
        print("\nPerformance Comparison:")
        print("=" * 60)
        print(f"{'Model':<20} {'Previous':<12} {'Phase 4':<12} {'Change':<12}")
        print("-" * 60)
        
        for model_name, current_acc in results_dict.items():
            if model_name in previous_results:
                prev_acc = previous_results[model_name]
                change = current_acc - prev_acc
                change_str = f"{change:+.4f}"
                print(f"{model_name:<20} {prev_acc:<12.4f} {current_acc:<12.4f} {change_str:<12}")
            else:
                print(f"{model_name:<20} {'N/A':<12} {current_acc:<12.4f} {'N/A':<12}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"⚠️ Could not load previous results: {e}")

def main():
    """Main training function"""
    print("=== UFC Model Training - Phase 4 ===")
    print("Training models with ELO features\n")
    
    # Load dataset
    df = load_elo_dataset()
    if df is None:
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # Train models
    models = {}
    results = {}
    
    # Decision Tree
    dt_model, dt_acc, dt_roc, dt_pred, dt_proba = train_decision_tree(X_train, X_test, y_train, y_test)
    models['Decision Tree'] = dt_model
    results['Decision Tree'] = dt_acc
    
    # Random Forest
    rf_model, rf_acc, rf_roc, rf_pred, rf_proba = train_random_forest(X_train, X_test, y_train, y_test)
    models['Random Forest'] = rf_model
    results['Random Forest'] = rf_acc
    
    # Optimized Random Forest
    rf_opt_model, rf_opt_acc, rf_opt_roc, rf_opt_pred, rf_opt_proba = optimize_random_forest(X_train, X_test, y_train, y_test)
    models['Random Forest Optimized'] = rf_opt_model
    results['Random Forest Optimized'] = rf_opt_acc
    
    # XGBoost
    xgb_model, xgb_acc, xgb_roc, xgb_pred, xgb_proba = train_xgboost(X_train, X_test, y_train, y_test)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = xgb_acc
    
    # Evaluate performance
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION")
    print("="*60)
    
    evaluate_model_performance(y_test, dt_pred, dt_proba, "Decision Tree")
    evaluate_model_performance(y_test, rf_pred, rf_proba, "Random Forest")
    evaluate_model_performance(y_test, rf_opt_pred, rf_opt_proba, "Random Forest Optimized")
    evaluate_model_performance(y_test, xgb_pred, xgb_proba, "XGBoost")
    
    # Analyze feature importance for best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    analyze_feature_importance(best_model, feature_cols, best_model_name)
    
    # Compare with previous models
    compare_with_previous_models(results)
    
    # Save models
    save_models(models)
    
    # Final summary
    print(f"\nPhase 4 Model Training Complete!")
    print(f"Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")
    print(f"Models saved to models/ directory")
    print(f"Performance plots saved to images/ directory")
    
    # Create completion marker
    with open("models/PHASE4_TRAINING_DONE.txt", "w") as f:
        f.write(f"Phase 4 Training Complete\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Best Accuracy: {results[best_model_name]:.4f}\n")
        f.write(f"Features Used: {len(feature_cols)}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
    
    print(f"✅ Training completion marker created")

if __name__ == "__main__":
    main()
