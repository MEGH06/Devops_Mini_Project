# -*- coding: utf-8 -*-
"""DevOps MiniProject with PyCaret

Diabetes Prediction using PyCaret AutoML
"""

import pandas as pd
from pycaret.classification import *

# Load the dataset
df = pd.read_csv(r"C:\Users\Megh\Desktop\3rd yr\projects\Devops_Mini Project\diabetes_dataset.csv")

print("="*50)
print("Dataset Overview")
print("="*50)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")

# =============================================================================
# PYCARET SETUP - This is where the magic happens!
# =============================================================================
print("\n" + "="*50)
print("Setting up PyCaret Environment")
print("="*50)

# PyCaret setup handles:
# - Train/test split
# - Data preprocessing (encoding, scaling, etc.)
# - Missing value imputation
# - Feature engineering
clf = setup(
    data=df,
    target='diagnosed_diabetes',
    
    # Categorical features (PyCaret will handle encoding automatically)
    categorical_features=['gender', 'ethnicity', 'education_level', 
                         'employment_status', 'income_level', 
                         'smoking_status', 'diabetes_stage'],
    
    # Data split configuration
    train_size=0.8,
    session_id=42,  # For reproducibility
    
    # Performance settings
    normalize=True,  # Normalize numerical features
    remove_outliers=False,
    fix_imbalance=False,  # Set to True if dataset is imbalanced
    
    # Display settings
    verbose=True,
    html=False  # Set to True if you want HTML report in notebook
)

# =============================================================================
# COMPARE MODELS - PyCaret will train multiple models and rank them
# =============================================================================
print("\n" + "="*50)
print("Comparing Multiple ML Models")
print("="*50)

# This trains 15+ models and ranks them by performance
best_models = compare_models(
    n_select=3,  # Select top 3 models
    sort='Accuracy',  # Can use 'AUC', 'Recall', 'Precision', 'F1'
)

# =============================================================================
# GET THE BEST MODEL
# =============================================================================
print("\n" + "="*50)
print("Best Model Details")
print("="*50)

# Get the single best model
best_model = best_models[0] if isinstance(best_models, list) else best_models
print(f"Best Model: {best_model}")

# =============================================================================
# TUNE THE BEST MODEL - Hyperparameter optimization
# =============================================================================
print("\n" + "="*50)
print("Tuning Best Model (Hyperparameter Optimization)")
print("="*50)

tuned_model = tune_model(
    best_model,
    n_iter=10,  # Number of iterations for tuning
    optimize='Accuracy'  # Metric to optimize
)

# =============================================================================
# EVALUATE MODEL - Detailed performance metrics
# =============================================================================
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

# This shows various plots: AUC, Confusion Matrix, Feature Importance, etc.
evaluate_model(tuned_model)

# =============================================================================
# PREDICTIONS ON TEST SET
# =============================================================================
print("\n" + "="*50)
print("Making Predictions on Test Set")
print("="*50)

# Make predictions on the test set
predictions = predict_model(tuned_model)
print(predictions.head())

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n" + "="*50)
print("Feature Importance Analysis")
print("="*50)

# Plot feature importance
plot_model(tuned_model, plot='feature')

# =============================================================================
# ADDITIONAL VISUALIZATIONS
# =============================================================================
print("\n" + "="*50)
print("Generating Additional Visualizations")
print("="*50)

# Confusion Matrix
plot_model(tuned_model, plot='confusion_matrix')

# AUC-ROC Curve
plot_model(tuned_model, plot='auc')

# Precision-Recall Curve
plot_model(tuned_model, plot='pr')

# Class Prediction Error
plot_model(tuned_model, plot='error')

# =============================================================================
# SAVE THE MODEL
# =============================================================================
print("\n" + "="*50)
print("Saving Model")
print("="*50)

# Save the trained model
save_model(tuned_model, 'diabetes_prediction_model')
print("✓ Model saved as 'diabetes_prediction_model.pkl'")

# =============================================================================
# LOAD AND USE MODEL (for deployment)
# =============================================================================
print("\n" + "="*50)
print("Example: Loading Saved Model for Predictions")
print("="*50)

# Load the model
loaded_model = load_model('diabetes_prediction_model')

# Make predictions on new data
# Example: predict on first 5 rows
sample_data = df.head(5).drop('diagnosed_diabetes', axis=1)
new_predictions = predict_model(loaded_model, data=sample_data)
print("\nPredictions on sample data:")
print(new_predictions[['prediction_label', 'prediction_score']])

print("\n" + "="*50)
print("Pipeline Complete!")
print("="*50)