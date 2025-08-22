#!/usr/bin/env python3
"""
UFC Model Training Implementation
Phase 3: Model Adaptation and Training
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_engineered_data():
    """Load the engineered UFC dataset"""
    print("Loading engineered UFC dataset...")
    df = pd.read_csv("dataUFC/ufc_engineered.csv")
    print(f"Dataset shape: {df.shape}")
    return df

def prepare_data(df):
    """Prepare data for training"""
    print("Preparing data for training...")
    
    # Select only numeric columns for training
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove non-numeric columns and target variable
    feature_columns = [col for col in numeric_columns if col != 'RESULT']
    
    print(f"Using {len(feature_columns)} numeric features for training")
    print(f"Feature columns: {feature_columns[:5]}...")  # Show first 5 features
    
    # Prepare feature data
    X = df[feature_columns].values
    y = df['RESULT'].values
    
    # Exclude first 1000 fights for stability
    X = X[1000:]
    y = y[1000:]
    
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split the data using 85% training, 15% testing
    split = 0.85
    value = round(split * len(X))
    
    x_train = X[:value]
    y_train = y[:value]
    x_test = X[value:]
    y_test = y[value:]
    
    print(f"Training Data: {x_train.shape}")
    print(f"Testing Data: {x_test.shape}")
    
    # Define mappers for target variable
    mapper = np.vectorize(lambda x: "Fighter2 Wins" if x == 0 else "Fighter1 Wins")
    
    # Convert target to string labels
    y_train_str = mapper(y_train)
    y_test_str = mapper(y_test)
    
    return x_train, y_train_str, x_test, y_test_str, feature_columns

def train_decision_tree(x_train, y_train, x_test, y_test, feature_names):
    """Train a simple decision tree"""
    print("\n=== Training Decision Tree ===")
    
    # Instantiate Decision Tree
    dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_model.fit(x_train, y_train)
    
    # Make predictions
    predictions_train = dt_model.predict(x_train)
    predictions_test = dt_model.predict(x_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, predictions_train)
    test_accuracy = accuracy_score(y_test, predictions_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Show tree structure
    text_representation = tree.export_text(dt_model, feature_names=feature_names, max_depth=3)
    print("\nDecision Tree Structure (first 3 levels):")
    print(text_representation[:1000] + "..." if len(text_representation) > 1000 else text_representation)
    
    return dt_model, train_accuracy, test_accuracy

def train_random_forest(x_train, y_train, x_test, y_test, feature_names):
    """Train Random Forest models"""
    print("\n=== Training Random Forest ===")
    
    # Model 1: Large Random Forest
    print("Training large Random Forest (n_estimators=500)...")
    rf_large = RandomForestClassifier(
        n_estimators=500, 
        max_depth=10, 
        max_features="sqrt", 
        bootstrap=True,
        random_state=42
    )
    rf_large.fit(x_train, y_train)
    
    predictions_train = rf_large.predict(x_train)
    predictions_test = rf_large.predict(x_test)
    
    train_accuracy_large = accuracy_score(y_train, predictions_train)
    test_accuracy_large = accuracy_score(y_test, predictions_test)
    
    print(f"Large RF - Train Accuracy: {train_accuracy_large:.4f}")
    print(f"Large RF - Test Accuracy: {test_accuracy_large:.4f}")
    
    # Model 2: Smaller, Less Overfitted Random Forest
    print("\nTraining smaller Random Forest (n_estimators=100)...")
    rf_small = RandomForestClassifier(
        n_estimators=100, 
        max_depth=7, 
        min_samples_split=50, 
        min_samples_leaf=25, 
        max_features="sqrt", 
        bootstrap=True,
        random_state=42
    )
    rf_small.fit(x_train, y_train)
    
    predictions_train = rf_small.predict(x_train)
    predictions_test = rf_small.predict(x_test)
    
    train_accuracy_small = accuracy_score(y_train, predictions_train)
    test_accuracy_small = accuracy_score(y_test, predictions_test)
    
    print(f"Small RF - Train Accuracy: {train_accuracy_small:.4f}")
    print(f"Small RF - Test Accuracy: {test_accuracy_small:.4f}")
    
    return rf_large, rf_small, train_accuracy_large, test_accuracy_large, train_accuracy_small, test_accuracy_small

def optimize_random_forest(x_train, y_train, x_test, y_test):
    """Optimize Random Forest hyperparameters"""
    print("\n=== Optimizing Random Forest Hyperparameters ===")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt', 'log2']
    }
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    print("Running Grid Search (this may take a while)...")
    grid_search.fit(x_train, y_train)
    
    # Best parameters
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Test best model
    best_rf = grid_search.best_estimator_
    predictions_test = best_rf.predict(x_test)
    test_accuracy = accuracy_score(y_test, predictions_test)
    print(f"Best Model Test Accuracy: {test_accuracy:.4f}")
    
    return best_rf, grid_search.best_params_, test_accuracy

def train_xgboost(x_train, y_train, x_test, y_test):
    """Train XGBoost model"""
    print("\n=== Training XGBoost ===")
    
    # Convert labels to numeric for XGBoost
    y_train_numeric = (y_train == "Fighter1 Wins").astype(int)
    y_test_numeric = (y_test == "Fighter1 Wins").astype(int)
    
    # XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_model.fit(x_train, y_train_numeric)
    
    # Make predictions
    predictions_train = xgb_model.predict(x_train)
    predictions_test = xgb_model.predict(x_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train_numeric, predictions_train)
    test_accuracy = accuracy_score(y_test_numeric, predictions_test)
    
    print(f"XGBoost - Train Accuracy: {train_accuracy:.4f}")
    print(f"XGBoost - Test Accuracy: {test_accuracy:.4f}")
    
    return xgb_model, train_accuracy, test_accuracy

def analyze_feature_importance(model, feature_names, model_name):
    """Analyze feature importance"""
    print(f"\n=== {model_name} Feature Importance ===")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("Top 15 Most Important Features:")
        for i in range(min(15, len(feature_names))):
            print(f"{i+1:2d}. {feature_names[indices[i]]:25s} - {importances[indices[i]]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} - Feature Importance')
        plt.bar(range(min(15, len(feature_names))), importances[indices[:15]])
        plt.xticks(range(min(15, len(feature_names))), 
                   [feature_names[i] for i in indices[:15]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'images/ufc_{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.close()
        
        return importances, indices
    else:
        print("Model doesn't have feature_importances_ attribute")
        return None, None

def evaluate_model_performance(model, x_test, y_test, model_name):
    """Evaluate model performance with detailed metrics"""
    print(f"\n=== {model_name} Performance Evaluation ===")
    
    # Make predictions
    predictions = model.predict(x_test)
    
    # Convert to numeric for ROC AUC
    y_test_numeric = (y_test == "Fighter1 Wins").astype(int)
    predictions_numeric = (predictions == "Fighter1 Wins").astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test_numeric, predictions_numeric)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fighter2 Wins', 'Fighter1 Wins'],
                yticklabels=['Fighter2 Wins', 'Fighter1 Wins'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'images/ufc_{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    
    return accuracy, roc_auc

def save_models(models_dict):
    """Save trained models"""
    print("\n=== Saving Models ===")
    
    for model_name, model in models_dict.items():
        filename = f"models/ufc_{model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, filename)
        print(f"Saved {model_name} to {filename}")

def main():
    """Main training function"""
    print("=== UFC Model Training Implementation ===\n")
    
    # Load data
    df = load_engineered_data()
    
    # Prepare data
    x_train, y_train, x_test, y_test, feature_names = prepare_data(df)
    
    # Train models
    models = {}
    results = {}
    
    # Decision Tree
    dt_model, dt_train_acc, dt_test_acc = train_decision_tree(x_train, y_train, x_test, y_test, feature_names)
    models['Decision Tree'] = dt_model
    results['Decision Tree'] = {'train_acc': dt_train_acc, 'test_acc': dt_test_acc}
    
    # Random Forest
    rf_large, rf_small, rf_large_train, rf_large_test, rf_small_train, rf_small_test = train_random_forest(
        x_train, y_train, x_test, y_test, feature_names
    )
    models['Random Forest Large'] = rf_large
    models['Random Forest Small'] = rf_small
    results['Random Forest Large'] = {'train_acc': rf_large_train, 'test_acc': rf_large_test}
    results['Random Forest Small'] = {'train_acc': rf_small_train, 'test_acc': rf_small_test}
    
    # Optimized Random Forest
    best_rf, best_params, best_rf_test_acc = optimize_random_forest(x_train, y_train, x_test, y_test)
    models['Random Forest Optimized'] = best_rf
    results['Random Forest Optimized'] = {'test_acc': best_rf_test_acc}
    
    # XGBoost
    xgb_model, xgb_train_acc, xgb_test_acc = train_xgboost(x_train, y_train, x_test, y_test)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {'train_acc': xgb_train_acc, 'test_acc': xgb_test_acc}
    
    # Decide best model
    best_model_name = max(results.keys(), key=lambda k: results[k].get('test_acc', 0))
    best_model = models[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    print(f"Test Accuracy: {results[best_model_name]['test_acc']:.4f}")

    # Save models EARLY so they're available even if plotting blocks
    save_models(models)

    # Also save best model separately for convenience
    import joblib as _joblib
    _best_path = f"models/ufc_best_model.joblib"
    _joblib.dump(best_model, _best_path)
    print(f"Saved Best Model to {_best_path}")

    # Feature importance analysis (non-blocking backend)
    analyze_feature_importance(best_model, feature_names, best_model_name)

    # Detailed performance evaluation (non-blocking backend)
    evaluate_model_performance(best_model, x_test, y_test, best_model_name)
    
    # Final summary
    print("\n=== Training Summary ===")
    print("Model Performance Comparison:")
    for model_name, result in results.items():
        test_acc = result.get('test_acc', 'N/A')
        train_acc = result.get('train_acc', 'N/A')
        print(f"{model_name:25s} - Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Test accuracy: {results[best_model_name]['test_acc']:.4f}")
    
    # Write DONE marker file with timestamp and best model
    import datetime, os
    done_path = os.path.join('models', 'TRAINING_DONE.txt')
    with open(done_path, 'w', encoding='utf-8') as f:
        f.write(f"Training completed at {datetime.datetime.now().isoformat()}\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write("Models saved to models/ directory\n")
    print(f"\nModel training completed successfully! DONE marker: {done_path}")
    print("Models saved to models/ directory")
    print("Visualizations saved to images/ directory")

if __name__ == "__main__":
    main()
