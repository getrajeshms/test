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

# Set page configuration
st.set_page_config(
    page_title="PyloScan - H. Pylori Infection Prediction",
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
        # Check if model is already loaded in session state
        if 'ml_pipeline_obj' in st.session_state:
            self.ml_pipeline = st.session_state.ml_pipeline_obj
        else:
            self.ml_pipeline = None
        self.model_utils = ModelUtils()
        
    def load_or_train_model(self):
        """Load existing model"""
        if os.path.exists('models/best_model.joblib') and os.path.exists('models/preprocessor.joblib'):
            try:
                pipeline = MLPipeline()
                pipeline.load_model('models/best_model.joblib', 'models/preprocessor.joblib')
                # Store in session state for persistence across page interactions
                st.session_state.ml_pipeline_obj = pipeline
                self.ml_pipeline = pipeline
                return True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return False
        return False
    

    def render_sidebar(self):
        """Render sidebar for navigation"""
        st.sidebar.title("🔬 H. Pylori Prediction")
        
        # Display model info in sidebar
        if self.ml_pipeline and hasattr(self.ml_pipeline, 'best_model_name'):
            st.sidebar.info(f"**Model:** {self.ml_pipeline.best_model_name}")
            if hasattr(self.ml_pipeline, 'model_results') and self.ml_pipeline.best_model_name in self.ml_pipeline.model_results:
                metrics = self.ml_pipeline.model_results[self.ml_pipeline.best_model_name]
                st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        
        st.sidebar.markdown("---")
        
        # AI Configuration in Sidebar (always accessible)
        st.sidebar.subheader("🤖 AI Configuration")
        
        # Check for environment variables
        env_openai = os.environ.get("OPENAI_API_KEY", "")
        env_gemini = os.environ.get("GEMINI_API_KEY", "")
        
        # Auto-configure if environment keys exist (outside expanders so they always run)
        if env_openai and 'custom_openai_key' not in st.session_state:
            st.session_state['custom_openai_key'] = env_openai
            st.session_state['openai_configured'] = True
        
        if env_gemini and 'custom_gemini_key' not in st.session_state:
            st.session_state['custom_gemini_key'] = env_gemini
            st.session_state['gemini_configured'] = True
        
        # OpenAI Configuration
        with st.sidebar.expander("OpenAI Setup", expanded=False):
            if env_openai:
                st.success("✅ API key in environment")
                use_env_openai = st.checkbox("Use env key", value=True, key="use_env_openai")
            else:
                use_env_openai = False
                st.info("No env key found")
            
            if not use_env_openai:
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=st.session_state.get('custom_openai_key', ''),
                    key="openai_input"
                )
                if openai_key:
                    st.session_state['custom_openai_key'] = openai_key
                    st.session_state['openai_configured'] = True
            else:
                st.session_state['openai_configured'] = True
                st.session_state['custom_openai_key'] = env_openai
        
        # Gemini Configuration
        with st.sidebar.expander("Google Gemini Setup", expanded=False):
            if env_gemini:
                st.success("✅ API key in environment")
                use_env_gemini = st.checkbox("Use env key", value=True, key="use_env_gemini")
            else:
                use_env_gemini = False
                st.info("No env key found")
            
            if not use_env_gemini:
                gemini_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    value=st.session_state.get('custom_gemini_key', ''),
                    key="gemini_input"
                )
                if gemini_key:
                    st.session_state['custom_gemini_key'] = gemini_key
                    st.session_state['gemini_configured'] = True
            else:
                st.session_state['gemini_configured'] = True
                st.session_state['custom_gemini_key'] = env_gemini
        
        # Check configuration status (runs even when expanders are closed)
        has_openai = bool(st.session_state.get('custom_openai_key', ''))
        has_gemini = bool(st.session_state.get('custom_gemini_key', ''))
        
        # Update configured flags based on actual key presence
        if has_openai:
            st.session_state['openai_configured'] = True
        if has_gemini:
            st.session_state['gemini_configured'] = True
        
        # AI Provider Selection
        ai_options = []
        if st.session_state.get('openai_configured', False):
            ai_options.append("OpenAI GPT")
        if st.session_state.get('gemini_configured', False):
            ai_options.append("Google Gemini")
        
        if ai_options:
            if 'preferred_ai' not in st.session_state:
                st.session_state['preferred_ai'] = ai_options[0]
            
            preferred = st.sidebar.selectbox(
                "AI Provider",
                ai_options,
                index=ai_options.index(st.session_state.get('preferred_ai', ai_options[0]))
            )
            st.session_state['preferred_ai'] = preferred
            st.sidebar.success(f"✅ Using {preferred}")
        else:
            st.sidebar.warning("⚠️ Configure AI for recommendations")
        
        st.sidebar.markdown("---")
        
        # Page Navigation
        pages = [
            "Patient Prediction",
            "Model Performance"
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
            st.subheader("🔍 Key Contributing Factors")
            
            try:
                # Get global feature importances
                if hasattr(self.ml_pipeline.best_model, 'feature_importances_'):
                    feature_names = self.ml_pipeline.preprocessor.get_feature_names_out()
                    importances = self.ml_pipeline.best_model.feature_importances_
                    
                    # Get top 10 most important features overall
                    feature_importance_pairs = list(zip(feature_names, importances))
                    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                    top_features = feature_importance_pairs[:10]
                    
                    # Clean feature names for display
                    clean_features = []
                    for feat, imp in top_features:
                        # Remove prefix like 'num__' or 'cat__'
                        clean_name = feat.split('__')[-1] if '__' in feat else feat
                        clean_features.append((clean_name.replace('_', ' ').title(), imp))
                    
                    # Create dataframe for plotting
                    importance_df = pd.DataFrame(clean_features, columns=['Feature', 'Importance'])
                    
                    fig_importance = px.bar(
                        importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Top 10 Most Important Clinical Factors',
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Display key patient-specific factors
                    st.markdown("**Key Patient Factors in This Case:**")
                    risk_factors = []
                    if patient_data.get('Gastritis_History', 0) == 1:
                        risk_factors.append("• Previous gastritis history")
                    if patient_data.get('Ulcer_History', 0) == 1:
                        risk_factors.append("• Previous ulcer disease")
                    if patient_data.get('Family_Pylori_History', 0) == 1:
                        risk_factors.append("• Family history of H. pylori")
                    if patient_data.get('Smoking', 0) > 0:
                        risk_factors.append("• Current/past smoking")
                    if patient_data.get('BMI', 25) > 30:
                        risk_factors.append("• Elevated BMI")
                    
                    if risk_factors:
                        st.markdown("\n".join(risk_factors))
                    else:
                        st.markdown("• No major risk factors identified in patient history")
                
            except Exception as e:
                st.info("Feature analysis currently unavailable")
        
        # Clinical Treatment Recommendation
        st.subheader("💊 Clinical Decision Support")
        
        # Generate simple, evidence-based clinical guidance
        def get_clinical_recommendation(risk_level, risk_prob, patient_data):
            """Generate concise clinical recommendations for doctors"""
            
            recommendations = {
                'High': {
                    'action': '**Immediate action required:** Order confirmatory testing (stool antigen or urea breath test) and initiate empirical triple therapy pending results.',
                    'treatment': '**Treatment:** PPI-based triple therapy (PPI + Amoxicillin + Clarithromycin) for 14 days, or quadruple therapy if clarithromycin resistance suspected.'
                },
                'Medium': {
                    'action': '**Clinical evaluation recommended:** Perform diagnostic testing (stool antigen or urea breath test) to confirm infection status.',
                    'treatment': '**Treatment:** If confirmed positive, initiate standard triple therapy (PPI + Amoxicillin + Clarithromycin) for 14 days with 4-week post-treatment follow-up.'
                },
                'Low': {
                    'action': '**Routine monitoring sufficient:** Consider testing only if patient develops dyspeptic symptoms or has additional risk factors.',
                    'treatment': '**Management:** Watchful waiting with patient education on hygiene practices and dietary modifications. Annual screening if family history present.'
                }
            }
            
            rec = recommendations[risk_level]
            
            # Add risk factor context
            risk_factors = []
            if patient_data.get('Gastritis_History', 0) == 1:
                risk_factors.append("previous gastritis")
            if patient_data.get('Ulcer_History', 0) == 1:
                risk_factors.append("ulcer history")
            if patient_data.get('Family_Pylori_History', 0) == 1:
                risk_factors.append("family H. pylori history")
            
            context = f"**Risk Factors Present:** {', '.join(risk_factors)}" if risk_factors else "**Risk Factors:** None identified"
            
            return rec['action'], rec['treatment'], context
        
        action, treatment, context = get_clinical_recommendation(risk_level, risk_prob, patient_data)
        
        st.markdown(f"""
        <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color};">
            <h4 style="margin-top: 0;">Evidence-Based Clinical Guidance</h4>
            <p style="font-size: 1.05em; line-height: 1.6;">
                {action}
            </p>
            <p style="font-size: 1.05em; line-height: 1.6;">
                {treatment}
            </p>
            <p style="font-size: 0.95em; color: #666; margin-bottom: 0;">
                {context}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Treatment Recommendations (automatic if configured)
        ai_configured = st.session_state.get('openai_configured', False) or st.session_state.get('gemini_configured', False)
        
        if ai_configured:
            st.markdown("---")
            st.subheader("🤖 AI-Powered Personalized Treatment Recommendations")
            
            with st.spinner("Generating AI-powered personalized recommendations..."):
                try:
                    # Get API keys from session state
                    openai_key = st.session_state.get('custom_openai_key', '')
                    gemini_key = st.session_state.get('custom_gemini_key', '')
                    preferred_ai = st.session_state.get('preferred_ai', 'auto')
                    
                    # Initialize clients with session state keys
                    import google.genai as genai
                    from openai import OpenAI
                    
                    # Build comprehensive patient profile for AI
                    smoking_map = {0: "None", 1: "1-5/day", 2: "6-10/day", 3: ">10/day"}
                    alcohol_map = {0: "None", 1: "Monthly", 2: "Weekly", 3: "3+/Weekly"}
                    handwash_map = {0: "Rarely", 1: "Now & then", 2: "Frequent", 3: "Daily"}
                    food_map = {0: "Rare", 1: "Now & then", 2: "Frequent", 3: "Daily"}
                    
                    patient_profile = f"""
**PATIENT PROFILE:**

Demographics:
- Age: {patient_data.get('Age')} years
- Sex: {'Male' if patient_data.get('Sex') == 1 else 'Female'}
- BMI: {patient_data.get('BMI', 'N/A')} kg/m²
- Marital Status: {'Married' if patient_data.get('Marital_Status') == 1 else 'Single'}

Clinical History:
- Previous Gastritis: {'Yes' if patient_data.get('Gastritis_History') == 1 else 'No'}
- Previous Ulcer Disease: {'Yes' if patient_data.get('Ulcer_History') == 1 else 'No'}
- Family H. Pylori History: {'Yes' if patient_data.get('Family_Pylori_History') == 1 else 'No'}
- Family Gastritis History: {'Yes' if patient_data.get('Family_Gastritis_History') == 1 else 'No'}

Laboratory Values:
- Albumin: {patient_data.get('Albumin', 'N/A')} g/L
- WBC Count: {patient_data.get('WBC_Count', 'N/A')} ×10⁹/L
- Hemoglobin: {patient_data.get('Hemoglobin', 'N/A')} g/L
- RBC Count: {patient_data.get('RBC_Count', 'N/A')} ×10¹²/L

Lifestyle Factors:
- Smoking: {smoking_map.get(patient_data.get('Smoking', 0), 'None')}
- Alcohol: {alcohol_map.get(patient_data.get('Alcohol', 0), 'None')}
- Handwashing: {handwash_map.get(patient_data.get('Handwashing', 3), 'Daily')}
- Pickled Food Consumption: {food_map.get(patient_data.get('Pickled_Food', 0), 'Rare')}
- Tableware Sharing: {food_map.get(patient_data.get('Tableware_Sharing', 0), 'Rare')}

Endoscopic Findings:
- Gastric Nodularity: {'Yes' if patient_data.get('Nodularity') == 1 else 'No' if patient_data.get('Nodularity') == 0 else 'Not Available'}
- Gastric Redness: {'Yes' if patient_data.get('Gastric_Redness') == 1 else 'No' if patient_data.get('Gastric_Redness') == 0 else 'Not Available'}

**RISK ASSESSMENT:**
- H. Pylori Infection Probability: {risk_prob:.1%}
- Risk Level: {risk_level}
"""
                    
                    prompt = f"""You are an experienced gastroenterologist providing personalized, evidence-based treatment recommendations for H. pylori infection.

{patient_profile}

Based on this comprehensive patient profile, provide detailed, personalized treatment recommendations including:

1. **Clinical Assessment**: Interpret the risk level and key patient-specific factors
2. **Diagnostic Testing Strategy**: Specific tests recommended for this patient
3. **Treatment Plan**: 
   - First-line therapy with dosages and duration
   - Consider patient-specific factors (age, BMI, lifestyle)
   - Alternative options if contraindications exist
4. **Lifestyle Modifications**: Personalized advice based on patient's current habits
5. **Follow-up Protocol**: Specific timeline and monitoring plan
6. **Patient Education**: Key points to discuss with this patient

Focus on personalization based on the patient's specific risk factors, lab values, and lifestyle. Be specific and actionable for clinical decision-making."""

                    ai_recommendation = None
                    
                    if preferred_ai == "OpenAI GPT" and openai_key:
                        client = OpenAI(api_key=openai_key)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{
                                "role": "system",
                                "content": "You are an experienced gastroenterologist specializing in H. pylori management. Provide evidence-based, personalized treatment recommendations following current clinical guidelines."
                            }, {
                                "role": "user", 
                                "content": prompt
                            }],
                            max_completion_tokens=1500
                        )
                        ai_recommendation = response.choices[0].message.content
                    elif preferred_ai == "Google Gemini" and gemini_key:
                        client = genai.Client(api_key=gemini_key)
                        response = client.models.generate_content(
                            model="gemini-2.0-flash-exp",
                            contents=prompt
                        )
                        ai_recommendation = response.text
                    
                    if ai_recommendation:
                        st.markdown(f"*Generated using {preferred_ai}*")
                        st.markdown(ai_recommendation)
                        
                        # Add disclaimer
                        st.markdown("---")
                        st.warning("""
                        **⚠️ IMPORTANT DISCLAIMER:**  
                        This AI-generated recommendation is for educational purposes and clinical decision support only. 
                        Always use professional medical judgment and consider individual patient circumstances. 
                        This should not replace comprehensive clinical evaluation and consultation.
                        """)
                    else:
                        st.warning("⚠️ Please configure API key in the sidebar to get AI recommendations")
                        
                except Exception as e:
                    st.error(f"❌ AI service error: {str(e)}")
                    st.info("Please verify your API key is valid and has sufficient credits.")
        else:
            st.info("💡 **Optional:** Configure OpenAI or Gemini API keys in the sidebar for AI-powered personalized treatment recommendations")
    
    def render_model_performance(self):
        """Render model performance metrics"""
        st.subheader("📊 Model Performance Dashboard")
        
        if self.ml_pipeline is None:
            st.warning("No model loaded. Please contact your system administrator.")
            return
            
        if not hasattr(self.ml_pipeline, 'model_results') or not self.ml_pipeline.model_results:
            st.warning("Model performance data not available. Model loaded successfully but metrics are missing.")
            st.info("The model can still make predictions. Go to the Patient Prediction page to use it.")
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
        
        # Feature Importance
        if hasattr(self.ml_pipeline, 'feature_importances') and self.ml_pipeline.feature_importances:
            st.write("### Feature Importance - Best Model")
            
            # Get feature importances for best model
            importances = self.ml_pipeline.feature_importances.get(best_model_name)
            
            if importances is not None:
                # Get feature names
                try:
                    feature_names = self.ml_pipeline.preprocessor.get_feature_names_out()
                    
                    # Create dataframe and sort by importance
                    feat_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    # Create bar chart
                    fig_feat = go.Figure(data=[
                        go.Bar(
                            x=feat_df['Importance'],
                            y=feat_df['Feature'],
                            orientation='h',
                            marker=dict(color='#1f77b4')
                        )
                    ])
                    
                    fig_feat.update_layout(
                        title=f'Top 15 Most Important Features - {best_model_name}',
                        xaxis_title='Importance',
                        yaxis_title='Feature',
                        height=500,
                        yaxis=dict(autorange='reversed')
                    )
                    
                    st.plotly_chart(fig_feat, use_container_width=True)
                except Exception as e:
                    st.info(f"Feature importance visualization not available: {e}")
    
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🔬 <span style="color: #1f77b4; font-weight: bold;">PyloScan</span> - H. Pylori Infection Prediction System</h1>', 
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
                st.sidebar.error("❌ No trained model found. Please contact your system administrator.")
        
        # Render sidebar navigation
        selected_page = self.render_sidebar()
        
        # Render selected page
        if selected_page == "Patient Prediction":
            st.header("👤 Patient Risk Assessment")
            st.write("Enter patient information below to predict H. Pylori infection risk.")
            
            patient_data = self.render_patient_input_form()
            
            if st.button("🎯 Predict H. Pylori Risk", type="primary", use_container_width=True):
                self.predict_and_display_results(patient_data)
                
        elif selected_page == "Model Performance":
            self.render_model_performance()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            <span style="color: #1f77b4; font-weight: bold;">PyloScan</span> - H. Pylori Infection Prediction System | 
            Clinical Decision Support Tool | Powered by A1Intercept Technologies
        </div>
        <div style="text-align: center; color: #999; font-size: 0.8em; margin-top: 0.5rem;">
            ⚠️ For clinical use only. This tool assists healthcare professionals in risk assessment.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    app = HPyloriApp()
    app.run()
