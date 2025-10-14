import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from data_generator import generate_synthetic_data
from ml_pipeline import MLPipeline
from ai_recommendations import get_ai_treatment_recommendation
from model_utils import ModelUtils
import shap
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="H. Pylori Infection Prediction System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class HPyloriApp:
    def __init__(self):
        self.ml_pipeline = None
        self.model_utils = ModelUtils()
        
    def load_or_train_model(self):
        """Load existing model or train new one"""
        if os.path.exists('models/best_model.joblib') and os.path.exists('models/preprocessor.joblib'):
            try:
                self.ml_pipeline = MLPipeline()
                self.ml_pipeline.load_model('models/best_model.joblib', 'models/preprocessor.joblib')
                return True
            except:
                return False
        return False
    
    def train_new_model(self):
        """Train a new model with synthetic data"""
        with st.spinner("Generating synthetic data and training models..."):
            # Generate synthetic data
            data = generate_synthetic_data(1000)
            
            # Initialize and train ML pipeline
            self.ml_pipeline = MLPipeline()
            results = self.ml_pipeline.train_models(data)
            
            # Save the best model
            os.makedirs('models', exist_ok=True)
            self.ml_pipeline.save_model('models/best_model.joblib', 'models/preprocessor.joblib')
            
            return results

    def render_sidebar(self):
        """Render sidebar for navigation"""
        st.sidebar.title("🔬 H. Pylori Prediction")
        
        pages = [
            "Patient Prediction",
            "Model Performance",
            "Data Generation & Training",
            "Feature Analysis"
        ]
        
        return st.sidebar.selectbox("Select Page", pages)
    
    def render_patient_input_form(self):
        """Render patient input form"""
        st.subheader("Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Demographics**")
            age = st.number_input("Age (years)", min_value=18, max_value=75, value=35)
            sex = st.selectbox("Sex", ["Male", "Female"])
            residence = st.selectbox("Residence", ["Urban", "County", "Suburban", "Village"])
            education = st.selectbox("Education Level", 
                                   ["Primary", "Secondary", "College", "Bachelor", "Postgraduate"])
            marital_status = st.selectbox("Marital Status", ["Unmarried", "Married"])
            
        with col2:
            st.write("**Clinical History & Symptoms**")
            bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=45.0, value=25.0, step=0.1)
            gastritis_history = st.selectbox("Non-atrophic gastritis history", ["No", "Yes"])
            ulcer_history = st.selectbox("Ulcer/Peptic ulcer history", ["No", "Yes"])
            
        with col3:
            st.write("**Laboratory Values**")
            albumin = st.number_input("Albumin (g/L)", min_value=25.0, max_value=55.0, value=40.0, step=0.1)
            wbc = st.number_input("WBC count (10⁹/L)", min_value=3.0, max_value=15.0, value=7.0, step=0.1)
            lymphocyte = st.number_input("Lymphocyte count (10⁹/L)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            neutrophil = st.number_input("Neutrophil count (10⁹/L)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
            rbc = st.number_input("RBC count (10¹²/L)", min_value=3.0, max_value=6.0, value=4.5, step=0.1)
            hemoglobin = st.number_input("Hemoglobin (g/L)", min_value=80.0, max_value=180.0, value=130.0, step=1.0)
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.write("**Lifestyle Factors**")
            smoking = st.selectbox("Smoking Status", ["None", "1-5/day", "6-10/day", ">10/day"])
            alcohol = st.selectbox("Alcohol Consumption", ["None", "Monthly", "Weekly", "3+/Weekly"])
            water_source = st.selectbox("Drinking Water Source", ["Tap", "Purified", "Mineral", "Other"])
            
        with col5:
            st.write("**Dietary Factors**")
            pickled_food = st.selectbox("Pickled food consumption", ["Rare", "Now & then", "Frequent", "Daily"])
            handwashing = st.selectbox("Handwashing frequency", ["Rarely", "Now & then", "Frequent", "Daily"])
            tableware_sharing = st.selectbox("Frequency of sharing tableware/utensils", 
                                           ["Rare", "Now & then", "Frequent", "Daily"])
            family_history = st.selectbox("Family history of pylori infection", ["No", "Yes"])
            gastritis_family = st.selectbox("Family history of gastritis", ["No", "Yes"])
            
        # Optional endoscopy features
        st.write("**Endoscopy/Imaging Features (Optional)**")
        col6, col7 = st.columns(2)
        
        with col6:
            nodularity = st.selectbox("Gastric nodularity on endoscopy", ["No", "Yes", "Not Available"])
            redness = st.selectbox("Gastric mucosal redness on endoscopy", ["No", "Yes", "Not Available"])
            
        # Create patient data dictionary
        patient_data = {
            'Age': age,
            'Sex': 1 if sex == "Male" else 0,
            'Residence': {"Urban": 0, "County": 1, "Suburban": 2, "Village": 3}[residence],
            'Education': {"Primary": 0, "Secondary": 1, "College": 2, "Bachelor": 3, "Postgraduate": 4}[education],
            'Marital_Status': 1 if marital_status == "Married" else 0,
            'Family_Size': 3,  # Default value
            'BMI': bmi,
            'Gastritis_History': 1 if gastritis_history == "Yes" else 0,
            'Ulcer_History': 1 if ulcer_history == "Yes" else 0,
            'Albumin': albumin,
            'WBC_Count': wbc,
            'Lymphocyte_Count': lymphocyte,
            'Neutrophil_Count': neutrophil,
            'RBC_Count': rbc,
            'Hemoglobin': hemoglobin,
            'Smoking': {"None": 0, "1-5/day": 1, "6-10/day": 2, ">10/day": 3}[smoking],
            'Alcohol': {"None": 0, "Monthly": 1, "Weekly": 2, "3+/Weekly": 3}[alcohol],
            'Water_Source': {"Tap": 0, "Purified": 1, "Mineral": 2, "Other": 3}[water_source],
            'Pickled_Food': {"Rare": 0, "Now & then": 1, "Frequent": 2, "Daily": 3}[pickled_food],
            'Handwashing': {"Rarely": 0, "Now & then": 1, "Frequent": 2, "Daily": 3}[handwashing],
            'Tableware_Sharing': {"Rare": 0, "Now & then": 1, "Frequent": 2, "Daily": 3}[tableware_sharing],
            'Family_Pylori_History': 1 if family_history == "Yes" else 0,
            'Family_Gastritis_History': 1 if gastritis_family == "Yes" else 0,
            'Nodularity': 1 if nodularity == "Yes" else 0 if nodularity == "No" else -1,
            'Gastric_Redness': 1 if redness == "Yes" else 0 if redness == "No" else -1
        }
        
        return patient_data
    
    def predict_and_display_results(self, patient_data):
        """Make prediction and display results"""
        if self.ml_pipeline is None:
            st.error("Model not loaded. Please train the model first.")
            return
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Make prediction
        prediction_proba = self.ml_pipeline.predict_proba(patient_df)[0]
        prediction = self.ml_pipeline.predict(patient_df)[0]
        
        # Determine risk level
        risk_prob = prediction_proba[1]  # Probability of infection
        if risk_prob >= 0.7:
            risk_level = "High"
            risk_class = "high-risk"
            risk_color = "#f44336"
        elif risk_prob >= 0.4:
            risk_level = "Medium"
            risk_class = "medium-risk"
            risk_color = "#ff9800"
        else:
            risk_level = "Low"
            risk_class = "low-risk"
            risk_color = "#4caf50"
        
        # Display results
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="prediction-box {risk_class}">
                <h3 style="margin: 0; color: {risk_color};">Risk Level: {risk_level}</h3>
                <p style="margin: 10px 0 0 0; font-size: 1.2em;">
                    Infection Probability: <strong>{risk_prob:.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Risk gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Confidence Interval</h4>
                <p>{risk_prob-0.1:.1%} - {risk_prob+0.1:.1%}</p>
                <h4>Recommendation</h4>
                <p>{"Immediate medical attention" if risk_level == "High" else 
                    "Regular monitoring" if risk_level == "Medium" else 
                    "Routine screening"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature importance for this prediction
        if hasattr(self.ml_pipeline, 'best_model') and self.ml_pipeline.best_model is not None:
            st.subheader("🔍 Feature Contribution Analysis")
            
            try:
                # Get feature importance
                feature_names = self.ml_pipeline.preprocessor.get_feature_names_out()
                X_processed = self.ml_pipeline.preprocessor.transform(patient_df)
                
                if hasattr(self.ml_pipeline.best_model, 'feature_importances_'):
                    importances = self.ml_pipeline.best_model.feature_importances_
                    
                    # Create feature importance plot
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=True).tail(10)
                    
                    fig_importance = px.bar(
                        importance_df, 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        title='Top 10 Feature Importance for Prediction',
                        color='importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate feature importance: {str(e)}")
        
        # AI Treatment Recommendation
        st.subheader("🤖 AI-Powered Treatment Recommendations")
        
        if st.button("Generate Treatment Recommendation", type="primary"):
            with st.spinner("Generating personalized treatment recommendation..."):
                try:
                    recommendation = get_ai_treatment_recommendation(
                        patient_data, risk_prob, risk_level
                    )
                    
                    st.markdown("### 💊 Personalized Treatment Plan")
                    st.markdown(recommendation)
                    
                except Exception as e:
                    st.error(f"Could not generate AI recommendation: {str(e)}")
                    st.info("Please ensure your API key is configured in the environment variables.")
    
    def render_model_performance(self):
        """Render model performance metrics"""
        st.subheader("📊 Model Performance Dashboard")
        
        if self.ml_pipeline is None or not hasattr(self.ml_pipeline, 'model_results'):
            st.warning("No model performance data available. Please train the model first.")
            return
        
        results = self.ml_pipeline.model_results
        
        # Display metrics table
        st.write("### Model Comparison")
        metrics_df = pd.DataFrame(results).T
        st.dataframe(metrics_df, use_container_width=True)
        
        # Best model visualization
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        st.write(f"### Best Model: {best_model_name}")
        
        best_metrics = results[best_model_name]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{best_metrics['accuracy']:.3f}")
        with col2:
            st.metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
        with col3:
            st.metric("Precision", f"{best_metrics['precision']:.3f}")
        with col4:
            st.metric("Recall", f"{best_metrics['recall']:.3f}")
        
        # ROC Curve comparison
        if hasattr(self.ml_pipeline, 'roc_curves'):
            st.write("### ROC Curve Comparison")
            fig_roc = go.Figure()
            
            for model_name, (fpr, tpr, auc) in self.ml_pipeline.roc_curves.items():
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc:.3f})'
                ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_roc.update_layout(
                title='ROC Curves Comparison',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800, height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Confusion Matrix for best model
        if hasattr(self.ml_pipeline, 'confusion_matrices'):
            st.write("### Confusion Matrix - Best Model")
            cm = self.ml_pipeline.confusion_matrices.get(best_model_name)
            if cm is not None:
                fig_cm = px.imshow(
                    cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Infection', 'Infection'],
                    y=['No Infection', 'Infection'],
                    text_auto=True,
                    aspect="auto",
                    title=f"Confusion Matrix - {best_model_name}"
                )
                st.plotly_chart(fig_cm, use_container_width=True)
    
    def render_data_training(self):
        """Render data generation and training interface"""
        st.subheader("🔬 Data Generation & Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Generate New Training Data")
            n_samples = st.number_input(
                "Number of samples to generate", 
                min_value=100, 
                max_value=5000, 
                value=1000, 
                step=100
            )
            
            if st.button("Generate Synthetic Data", type="primary"):
                with st.spinner("Generating synthetic data..."):
                    data = generate_synthetic_data(n_samples)
                    st.success(f"Generated {len(data)} synthetic patient records!")
                    
                    # Show data preview
                    st.write("### Data Preview")
                    st.dataframe(data.head(), use_container_width=True)
                    
                    # Show data statistics
                    st.write("### Data Statistics")
                    col_stats1, col_stats2 = st.columns(2)
                    
                    with col_stats1:
                        st.write("**Target Distribution**")
                        target_dist = data['H_Pylori_Infection'].value_counts()
                        fig_target = px.pie(
                            values=target_dist.values,
                            names=['No Infection', 'Infection'],
                            title='H. Pylori Infection Distribution'
                        )
                        st.plotly_chart(fig_target, use_container_width=True)
                    
                    with col_stats2:
                        st.write("**Age Distribution**")
                        fig_age = px.histogram(
                            data, x='Age', 
                            title='Age Distribution',
                            nbins=20
                        )
                        st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.write("### Train ML Models")
            
            if st.button("Train New Models", type="secondary"):
                results = self.train_new_model()
                st.success("Model training completed!")
                
                # Display results
                st.write("### Training Results")
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df, use_container_width=True)
                
                # Plot comparison
                fig_comparison = px.bar(
                    x=list(results.keys()),
                    y=[results[model]['roc_auc'] for model in results.keys()],
                    title='Model Performance Comparison (ROC-AUC)',
                    labels={'x': 'Model', 'y': 'ROC-AUC Score'}
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    def render_feature_analysis(self):
        """Render feature analysis"""
        st.subheader("🔍 Feature Analysis & Data Insights")
        
        # Generate sample data for analysis
        if st.button("Load Sample Data for Analysis"):
            with st.spinner("Loading data..."):
                data = generate_synthetic_data(1000)
                
                st.write("### Correlation Analysis")
                
                # Select numeric columns for correlation
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                corr_matrix = data[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Feature distributions by target
                st.write("### Feature Distributions by H. Pylori Status")
                
                key_features = ['Age', 'BMI', 'Albumin', 'WBC_Count', 'Hemoglobin']
                
                for feature in key_features:
                    if feature in data.columns:
                        fig_dist = px.box(
                            data, 
                            x='H_Pylori_Infection', 
                            y=feature,
                            title=f'{feature} Distribution by H. Pylori Status',
                            labels={'H_Pylori_Infection': 'H. Pylori Status'}
                        )
                        fig_dist.update_xaxis(
                            tickmode='array',
                            tickvals=[0, 1],
                            ticktext=['No Infection', 'Infection']
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🔬 H. Pylori Infection Prediction System</h1>', 
                   unsafe_allow_html=True)
        
        # Initialize session state
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        
        # Load model on startup
        if not st.session_state.model_loaded:
            if self.load_or_train_model():
                st.session_state.model_loaded = True
                st.sidebar.success("✅ Model loaded successfully!")
            else:
                st.sidebar.warning("⚠️ No trained model found. Please train a new model.")
        
        # Render sidebar navigation
        selected_page = self.render_sidebar()
        
        # Render selected page
        if selected_page == "Patient Prediction":
            st.header("👤 Patient Risk Assessment")
            
            patient_data = self.render_patient_input_form()
            
            if st.button("🎯 Predict H. Pylori Risk", type="primary", use_container_width=True):
                self.predict_and_display_results(patient_data)
                
        elif selected_page == "Model Performance":
            self.render_model_performance()
            
        elif selected_page == "Data Generation & Training":
            self.render_data_training()
            
        elif selected_page == "Feature Analysis":
            self.render_feature_analysis()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            H. Pylori Infection Prediction System | 
            Built with Streamlit, Scikit-learn, and AI-powered recommendations
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = HPyloriApp()
    app.run()
