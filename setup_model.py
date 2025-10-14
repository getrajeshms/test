#!/usr/bin/env python3
"""
H. Pylori Prediction Model - One-Time Setup Script

This script performs the one-time setup for the H. Pylori infection prediction system:
1. Generates synthetic training data
2. Trains and validates multiple ML models
3. Saves the best performing model

Run this script only once during initial setup or when you need to retrain the model.
"""

from data_generator import generate_synthetic_data
from ml_pipeline import MLPipeline
import os
import pandas as pd

def main():
    print('='*70)
    print('H. PYLORI PREDICTION MODEL - ONE-TIME SETUP')
    print('='*70)
    
    # Step 1: Generate synthetic training data
    print('\n📊 Step 1: Generating synthetic patient data...')
    print('-' * 70)
    
    n_samples = 1000
    data = generate_synthetic_data(n_samples)
    
    print(f'✓ Generated {len(data)} patient records')
    print(f'✓ Features: {data.shape[1]} columns')
    print(f'✓ H. Pylori infection prevalence: {data["H_Pylori_Infection"].mean():.1%}')
    
    # Save the training dataset
    os.makedirs('data', exist_ok=True)
    data.to_csv('data/training_data.csv', index=False)
    print(f'✓ Training data saved to: data/training_data.csv')
    
    # Step 2: Train machine learning models
    print('\n🤖 Step 2: Training machine learning models...')
    print('-' * 70)
    
    pipeline = MLPipeline()
    results = pipeline.train_models(data)
    
    # Step 3: Display results
    print('\n📈 Step 3: Model Performance Summary')
    print('-' * 70)
    print(f'{"Model":<25} {"Accuracy":<12} {"Precision":<12} {"Recall":<12} {"ROC-AUC":<12}')
    print('-' * 70)
    
    for model_name, metrics in results.items():
        print(f'{model_name:<25} {metrics["accuracy"]:<12.3f} {metrics["precision"]:<12.3f} '
              f'{metrics["recall"]:<12.3f} {metrics["roc_auc"]:<12.3f}')
    
    # Step 4: Save the best model
    print('\n💾 Step 4: Saving the best model...')
    print('-' * 70)
    
    best_model_name = pipeline.best_model_name
    best_roc_auc = results[best_model_name]['roc_auc']
    
    os.makedirs('models', exist_ok=True)
    pipeline.save_model('models/best_model.joblib', 'models/preprocessor.joblib')
    
    print(f'✓ Best Model: {best_model_name}')
    print(f'✓ ROC-AUC Score: {best_roc_auc:.3f}')
    print(f'✓ Model saved to: models/best_model.joblib')
    print(f'✓ Preprocessor saved to: models/preprocessor.joblib')
    
    # Step 5: Test prediction
    print('\n🧪 Step 5: Testing prediction...')
    print('-' * 70)
    
    test_patient = data.iloc[[0]].drop('H_Pylori_Infection', axis=1)
    prediction_proba = pipeline.predict_proba(test_patient)
    prediction = pipeline.predict(test_patient)
    
    print(f'✓ Test Patient Prediction:')
    print(f'  - No Infection Probability: {prediction_proba[0][0]:.1%}')
    print(f'  - Infection Probability: {prediction_proba[0][1]:.1%}')
    print(f'  - Predicted Class: {"Infected" if prediction[0] == 1 else "Not Infected"}')
    
    # Summary
    print('\n' + '='*70)
    print('✅ SETUP COMPLETED SUCCESSFULLY!')
    print('='*70)
    print('\nThe H. Pylori prediction system is now ready for use.')
    print('\nNext Steps:')
    print('1. Start the Streamlit app: streamlit run app.py --server.port 5000')
    print('2. Navigate to the "AI Settings" page to configure your API keys')
    print('3. Use the "Patient Prediction" page to predict infection risk')
    print('\nFor more information, see README.md')
    print('='*70)

if __name__ == "__main__":
    main()
