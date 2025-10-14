# H. Pylori Infection Prediction System

A clinical decision support tool powered by Machine Learning and AI for predicting H. Pylori infection risk in patients.

## 🎯 Overview

This application uses advanced machine learning algorithms to predict the probability of H. Pylori infection based on patient demographics, clinical history, laboratory values, and lifestyle factors. It provides:

- **Risk Prediction**: Probability scores and risk categorization (Low/Medium/High)
- **AI-Powered Treatment Recommendations**: Personalized treatment plans using OpenAI GPT or Google Gemini
- **Model Performance Metrics**: Transparent view of model accuracy and reliability
- **Clinical Decision Support**: Evidence-based recommendations following medical guidelines

## 📊 Model Performance

The system uses a **Random Forest** classifier trained on 1,000 synthetic patient records:

- **ROC-AUC Score**: 85.2%
- **Accuracy**: 78.5%
- **Precision**: 71.2%
- **Recall**: 66.2%
- **Cross-validation ROC-AUC**: 90.5% (± 4.1%)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Google Gemini API key (for AI recommendations)

### Setup Instructions

The system has already been set up with:
1. ✅ Synthetic training data generated (1,000 patient records)
2. ✅ Model trained and validated
3. ✅ Best model saved and ready to use

### Running the Application

The application is already running! Access it at:
```
http://localhost:5000
```

## 📝 How to Use

### 1. **Patient Prediction Page** (Main Page)
   - Enter patient demographic information
   - Input clinical history and symptoms
   - Provide laboratory values
   - Add lifestyle and family history data
   - Click "Predict H. Pylori Risk"
   - View risk probability and category
   - Generate AI-powered treatment recommendations

### 2. **Model Performance Page**
   - View comprehensive model metrics
   - Compare different ML algorithms
   - Analyze ROC curves and confusion matrices
   - Understand model reliability

### 3. **AI Settings Page**
   - Configure OpenAI or Google Gemini API keys
   - Select preferred AI provider
   - Test connection to ensure proper setup

## 🔬 Features Included

### Patient Data Input
- **Demographics**: Age, Sex, Residence, Education, Marital Status, Family Size
- **Clinical History**: BMI, Gastritis History, Ulcer History
- **Laboratory Values**: Albumin, WBC Count, Lymphocyte, Neutrophil, RBC, Hemoglobin
- **Lifestyle Factors**: Smoking, Alcohol, Water Source, Pickled Food, Handwashing, Tableware Sharing
- **Family History**: H. Pylori infection, Gastritis
- **Endoscopy Findings**: Gastric Nodularity, Mucosal Redness (optional)

### Model Features
- **Feature Engineering**: 35 engineered features from 26 original inputs
- **Advanced ML Pipeline**: Multiple algorithms compared (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM)
- **Class Balancing**: SMOTE applied for handling class imbalance
- **Cross-Validation**: 5-fold stratified validation for robust performance

### AI Recommendations
- **Evidence-Based**: Following ACG, AGA, and Maastricht VI guidelines
- **Personalized**: Tailored to patient risk profile and clinical features
- **Comprehensive**: Includes diagnostic tests, treatment plans, lifestyle modifications, and follow-up care

## 📂 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── data_generator.py           # Synthetic data generation
├── ml_pipeline.py              # ML model training and prediction
├── ai_recommendations.py       # AI-powered treatment recommendations
├── model_utils.py              # Visualization and utilities
├── data/
│   └── training_data.csv       # Training dataset (1,000 patients)
└── models/
    ├── best_model.joblib       # Trained Random Forest model
    ├── preprocessor.joblib     # Feature preprocessing pipeline
    └── best_model_metadata.joblib  # Model metadata
```

## 🔑 Configuring AI Recommendations

### Option 1: Using the UI (Recommended)
1. Navigate to **AI Settings** page in the app
2. Enter your API key for OpenAI or Google Gemini
3. Click "Test AI Connection" to verify
4. Go back to Patient Prediction and generate recommendations

### Option 2: Using Environment Variables
Set environment variables before starting the app:
```bash
export OPENAI_API_KEY="your-openai-api-key"
# OR
export GEMINI_API_KEY="your-gemini-api-key"
```

### Getting API Keys
- **OpenAI**: https://platform.openai.com/api-keys
- **Google Gemini**: https://aistudio.google.com/apikey

## ⚠️ Important Notes

### Clinical Use
- This tool is designed for **clinical decision support only**
- It assists healthcare professionals in risk assessment
- **Should not replace professional medical judgment**
- Always consult with qualified healthcare providers for diagnosis and treatment

### Data Privacy
- No patient data is stored permanently
- API keys are stored only in the session (not saved to disk)
- For production use, implement proper security measures

### Model Limitations
- Trained on synthetic data simulating real-world patterns
- Performance may vary with actual patient populations
- Regular validation with real data is recommended
- Consider local antibiotic resistance patterns for treatment

## 🛠️ Technical Details

### Machine Learning Pipeline
1. **Data Preprocessing**: Standardization, missing value handling
2. **Feature Engineering**: Derived features, interaction terms, risk scores
3. **Model Training**: Multiple algorithms with hyperparameter tuning
4. **Validation**: Cross-validation, ROC-AUC, confusion matrices
5. **Prediction**: Real-time inference with probability calibration

### Supported Models
- Logistic Regression
- Random Forest (currently selected as best)
- XGBoost
- Gradient Boosting
- Support Vector Machine

### Technologies Used
- **Framework**: Streamlit
- **ML Libraries**: Scikit-learn, XGBoost, Imbalanced-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **AI Integration**: OpenAI GPT-5, Google Gemini 2.5 Pro

## 📈 Future Enhancements

Potential features for future versions:
- Batch prediction for multiple patients via CSV upload
- Model retraining pipeline with new patient data
- Patient history tracking and longitudinal analysis
- PDF export for prediction reports
- Real-time model performance monitoring

## 📄 License

This is a clinical decision support tool for educational and research purposes.

## 👨‍⚕️ For Healthcare Professionals

This tool provides:
- **Risk Stratification**: Identify high-risk patients for targeted intervention
- **Evidence-Based Guidance**: Treatment recommendations following international guidelines
- **Clinical Context**: Interpretable predictions with feature importance
- **Workflow Integration**: Easy-to-use interface for busy clinical settings

---

**Disclaimer**: This application is provided as-is for educational purposes. Always consult with qualified medical professionals for patient care decisions.
