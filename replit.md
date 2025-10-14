# H. Pylori Infection Prediction System

## Project Overview

A clinical decision support tool for predicting H. Pylori infection risk using Machine Learning and AI-powered treatment recommendations.

## Current Implementation Status

### ✅ Completed Features

1. **Synthetic Data Generation** (One-Time Setup)
   - Generated 1,000 realistic patient records
   - 26 clinical features across demographics, history, labs, and lifestyle
   - Saved to `data/training_data.csv`

2. **Machine Learning Pipeline**
   - Trained 5 classification models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, SVM)
   - Best model: Random Forest with 85.2% ROC-AUC
   - Feature engineering: 35 engineered features
   - SMOTE balancing for class imbalance
   - Saved to `models/` directory

3. **Streamlit Web Application**
   - Patient prediction interface with comprehensive input forms
   - Risk stratification (Low/Medium/High)
   - Model performance dashboard
   - AI-powered treatment recommendations
   - API key configuration via UI

4. **AI Integration**
   - Support for OpenAI GPT-5 and Google Gemini 2.5 Pro
   - Evidence-based treatment recommendations
   - Following ACG, AGA, and Maastricht VI guidelines
   - Configurable via AI Settings page

## Application Structure

### Pages
1. **Patient Prediction** (Main): Enter patient data and get risk predictions
2. **Model Performance**: View model metrics, ROC curves, confusion matrices
3. **AI Settings**: Configure API keys for AI recommendations

### Key Files
- `app.py` - Main Streamlit application
- `data_generator.py` - Synthetic data generation
- `ml_pipeline.py` - ML training and prediction pipeline
- `ai_recommendations.py` - AI treatment recommendations
- `model_utils.py` - Visualization utilities
- `setup_model.py` - One-time setup script

## Model Performance

**Random Forest Classifier:**
- ROC-AUC: 85.2%
- Accuracy: 78.5%
- Precision: 71.2%
- Recall: 66.2%
- Cross-validation ROC-AUC: 90.5% (± 4.1%)

## Usage Instructions

### For End Users (Doctors/Clinicians)
1. Open the application (already running on port 5000)
2. Go to **AI Settings** to configure API keys (if needed)
3. Navigate to **Patient Prediction** page
4. Enter patient information in the form
5. Click "Predict H. Pylori Risk"
6. View risk probability and generate AI treatment recommendations

### For Administrators

**Initial Setup (Already Done):**
```bash
python setup_model.py
```

**Running the Application:**
```bash
streamlit run app.py --server.port 5000
```

## Data Flow

1. **Input**: Patient demographics, clinical history, lab values, lifestyle factors
2. **Processing**: Feature engineering (35 features), standardization
3. **Prediction**: Random Forest classifier outputs probability
4. **Risk Categorization**: 
   - Low: < 40%
   - Medium: 40-70%
   - High: ≥ 70%
5. **AI Recommendations**: Personalized treatment plan based on risk and patient profile

## API Keys Required

For AI treatment recommendations, configure one of:
- **OpenAI API Key**: Get from https://platform.openai.com/api-keys
- **Gemini API Key**: Get from https://aistudio.google.com/apikey

Keys can be configured via:
- UI: AI Settings page (session-based)
- Environment variables: `OPENAI_API_KEY` or `GEMINI_API_KEY`

## Important Notes

### Clinical Use
- ⚠️ For clinical decision support only
- Not a replacement for professional medical judgment
- Assists healthcare professionals in risk assessment

### Data & Privacy
- No patient data stored permanently
- Model trained on synthetic data
- API keys stored only in session (not persisted)

### Model Limitations
- Trained on synthetic data (simulates real patterns)
- Performance may vary with actual patient populations
- Regular validation with real data recommended

## Recent Changes (October 2025)

1. **Simplified Application**
   - Removed data generation UI (one-time setup only)
   - Removed model training UI (pre-trained model)
   - Focused on patient prediction workflow

2. **Enhanced AI Settings**
   - Added UI for API key configuration
   - Test connection feature
   - Support for both OpenAI and Gemini

3. **Model Optimization**
   - Selected Random Forest as best performer
   - Saved trained model for production use
   - No retraining needed during app usage

## Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **AI Services**: OpenAI GPT-5, Google Gemini 2.5 Pro
- **Data Balancing**: Imbalanced-learn (SMOTE)

## Future Enhancements (Planned)

1. Batch prediction via CSV upload
2. Model retraining pipeline with new data
3. Patient history tracking
4. PDF report export
5. Real-time performance monitoring
