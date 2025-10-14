import os
import json
from typing import Dict, Any, Optional
import google.genai as genai
from google.genai import types
from openai import OpenAI

# Initialize AI clients
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

openai_client = None
gemini_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class AIRecommendationEngine:
    """
    AI-powered treatment recommendation system for H. Pylori infection
    """
    
    def __init__(self):
        self.medical_guidelines = {
            'high_risk': {
                'urgency': 'Immediate medical attention required',
                'testing': ['Stool antigen test', 'Urea breath test', 'Upper endoscopy'],
                'treatment_approach': 'Triple or quadruple therapy',
                'followup': '4-6 weeks post-treatment testing'
            },
            'medium_risk': {
                'urgency': 'Medical evaluation recommended',
                'testing': ['Stool antigen test', 'Urea breath test'],
                'treatment_approach': 'Consider empirical treatment if positive',
                'followup': 'Monitor symptoms and retest if needed'
            },
            'low_risk': {
                'urgency': 'Routine screening adequate',
                'testing': ['Consider testing only if symptomatic'],
                'treatment_approach': 'Watchful waiting',
                'followup': 'Annual screening if risk factors present'
            }
        }
    
    def create_patient_profile(self, patient_data: Dict[str, Any], 
                             risk_probability: float, risk_level: str) -> str:
        """
        Create a comprehensive patient profile for AI analysis
        """
        # Convert numeric codes to readable text
        sex_text = "Male" if patient_data.get('Sex', 0) == 1 else "Female"
        
        residence_map = {0: "Urban", 1: "County", 2: "Suburban", 3: "Village"}
        residence_text = residence_map.get(patient_data.get('Residence', 0), "Unknown")
        
        education_map = {0: "Primary", 1: "Secondary", 2: "College", 3: "Bachelor", 4: "Postgraduate"}
        education_text = education_map.get(patient_data.get('Education', 0), "Unknown")
        
        smoking_map = {0: "None", 1: "1-5/day", 2: "6-10/day", 3: ">10/day"}
        smoking_text = smoking_map.get(patient_data.get('Smoking', 0), "None")
        
        alcohol_map = {0: "None", 1: "Monthly", 2: "Weekly", 3: "3+/Weekly"}
        alcohol_text = alcohol_map.get(patient_data.get('Alcohol', 0), "None")
        
        profile = f"""
        PATIENT CLINICAL PROFILE:
        
        Demographics:
        - Age: {patient_data.get('Age', 'Unknown')} years
        - Sex: {sex_text}
        - Residence: {residence_text}
        - Education: {education_text}
        - Marital Status: {"Married" if patient_data.get('Marital_Status', 0) == 1 else "Unmarried"}
        
        Clinical History:
        - BMI: {patient_data.get('BMI', 'Unknown')} kg/m²
        - Previous Gastritis: {"Yes" if patient_data.get('Gastritis_History', 0) == 1 else "No"}
        - Previous Ulcer Disease: {"Yes" if patient_data.get('Ulcer_History', 0) == 1 else "No"}
        
        Laboratory Values:
        - Albumin: {patient_data.get('Albumin', 'Unknown')} g/L
        - WBC Count: {patient_data.get('WBC_Count', 'Unknown')} ×10⁹/L
        - Lymphocyte Count: {patient_data.get('Lymphocyte_Count', 'Unknown')} ×10⁹/L
        - Neutrophil Count: {patient_data.get('Neutrophil_Count', 'Unknown')} ×10⁹/L
        - RBC Count: {patient_data.get('RBC_Count', 'Unknown')} ×10¹²/L
        - Hemoglobin: {patient_data.get('Hemoglobin', 'Unknown')} g/L
        
        Lifestyle Factors:
        - Smoking: {smoking_text}
        - Alcohol Consumption: {alcohol_text}
        - Handwashing Frequency: {["Rarely", "Now & then", "Frequent", "Daily"][patient_data.get('Handwashing', 3)]}
        - Pickled Food Consumption: {["Rare", "Now & then", "Frequent", "Daily"][patient_data.get('Pickled_Food', 0)]}
        - Tableware Sharing: {["Rare", "Now & then", "Frequent", "Daily"][patient_data.get('Tableware_Sharing', 0)]}
        
        Family History:
        - Family H. Pylori History: {"Yes" if patient_data.get('Family_Pylori_History', 0) == 1 else "No"}
        - Family Gastritis History: {"Yes" if patient_data.get('Family_Gastritis_History', 0) == 1 else "No"}
        
        RISK ASSESSMENT:
        - H. Pylori Infection Probability: {risk_probability:.1%}
        - Risk Level: {risk_level.upper()}
        
        ENDOSCOPIC FINDINGS (if available):
        - Gastric Nodularity: {"Yes" if patient_data.get('Nodularity', 0) == 1 else "No" if patient_data.get('Nodularity', 0) == 0 else "Not Available"}
        - Gastric Mucosal Redness: {"Yes" if patient_data.get('Gastric_Redness', 0) == 1 else "No" if patient_data.get('Gastric_Redness', 0) == 0 else "Not Available"}
        """
        
        return profile
    
    def get_openai_recommendation(self, patient_profile: str, risk_level: str) -> str:
        """
        Get treatment recommendation using OpenAI GPT
        """
        if not openai_client:
            raise ValueError("OpenAI API key not configured")
        
        system_prompt = """
        You are an experienced gastroenterologist and infectious disease specialist. 
        Provide evidence-based, personalized treatment recommendations for H. pylori infection 
        based on current medical guidelines (ACG, AGA, Maastricht VI consensus).
        
        Structure your response with:
        1. Risk Assessment Summary
        2. Recommended Diagnostic Tests
        3. Treatment Plan (if indicated)
        4. Lifestyle Modifications
        5. Follow-up Care
        6. Patient Education Points
        
        Be specific, actionable, and consider patient-specific factors.
        Include contraindications and alternative approaches when relevant.
        """
        
        user_prompt = f"""
        Based on the following patient profile, provide comprehensive treatment recommendations:
        
        {patient_profile}
        
        Consider the risk level ({risk_level}) and provide evidence-based recommendations 
        following current gastroenterology guidelines.
        """
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=2048
        )
        
        return response.choices[0].message.content
    
    def get_gemini_recommendation(self, patient_profile: str, risk_level: str) -> str:
        """
        Get treatment recommendation using Google Gemini
        """
        if not gemini_client:
            raise ValueError("Gemini API key not configured")
        
        prompt = f"""
        You are an experienced gastroenterologist providing evidence-based treatment recommendations.
        
        Based on the following patient profile, provide comprehensive H. pylori management recommendations:
        
        {patient_profile}
        
        Please structure your response with:
        1. **Risk Assessment Summary**
        2. **Recommended Diagnostic Tests**
        3. **Treatment Plan** (if indicated)
        4. **Lifestyle Modifications**
        5. **Follow-up Care**
        6. **Patient Education Points**
        
        Follow current medical guidelines (ACG, AGA, Maastricht VI) and consider:
        - Patient-specific risk factors
        - Contraindications and drug interactions
        - Local resistance patterns (assume standard patterns)
        - Cost-effectiveness
        
        Risk Level: {risk_level}
        """
        
        response = gemini_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        
        return response.text
    
    def add_clinical_guidelines(self, risk_level: str) -> str:
        """
        Add evidence-based clinical guidelines to the recommendation
        """
        guidelines = self.medical_guidelines.get(risk_level.lower(), {})
        
        guideline_text = f"""
        
        ## CLINICAL GUIDELINES ({risk_level.upper()} RISK)
        
        **Urgency**: {guidelines.get('urgency', 'Standard care')}
        
        **Recommended Testing**: {', '.join(guidelines.get('testing', ['Standard H. pylori testing']))}
        
        **Treatment Approach**: {guidelines.get('treatment_approach', 'Standard guidelines')}
        
        **Follow-up**: {guidelines.get('followup', 'As clinically indicated')}
        
        ## EVIDENCE-BASED TREATMENT OPTIONS
        
        **First-Line Therapy (Triple Therapy)**:
        - PPI + Amoxicillin + Clarithromycin (14 days) - if clarithromycin resistance <15%
        - PPI + Amoxicillin + Metronidazole (14 days) - alternative
        
        **Quadruple Therapy (Higher Efficacy)**:
        - PPI + Bismuth + Metronidazole + Tetracycline (14 days)
        - Concomitant: PPI + Amoxicillin + Clarithromycin + Metronidazole (14 days)
        
        **Second-Line Options**:
        - Levofloxacin-based triple therapy
        - Rifabutin-based therapy (refractory cases)
        
        ## IMPORTANT CONSIDERATIONS
        - Always test for cure 4-8 weeks post-treatment
        - Consider local antibiotic resistance patterns
        - Screen for drug allergies and contraindications
        - Patient compliance is crucial for success
        """
        
        return guideline_text

def get_ai_treatment_recommendation(patient_data: Dict[str, Any], 
                                  risk_probability: float, 
                                  risk_level: str,
                                  preferred_ai: str = "auto") -> str:
    """
    Get comprehensive AI-powered treatment recommendation
    
    Args:
        patient_data: Dictionary containing patient information
        risk_probability: Predicted probability of H. pylori infection
        risk_level: Risk category (Low/Medium/High)
        preferred_ai: "openai", "gemini", or "auto"
    
    Returns:
        str: Comprehensive treatment recommendation
    """
    
    engine = AIRecommendationEngine()
    
    # Create patient profile
    patient_profile = engine.create_patient_profile(
        patient_data, risk_probability, risk_level
    )
    
    # Get AI recommendation
    try:
        if preferred_ai == "openai" or (preferred_ai == "auto" and openai_client):
            ai_recommendation = engine.get_openai_recommendation(patient_profile, risk_level)
            ai_source = "OpenAI GPT-5"
        elif preferred_ai == "gemini" or (preferred_ai == "auto" and gemini_client):
            ai_recommendation = engine.get_gemini_recommendation(patient_profile, risk_level)
            ai_source = "Google Gemini"
        else:
            raise ValueError("No AI service available")
        
        # Add clinical guidelines
        clinical_guidelines = engine.add_clinical_guidelines(risk_level)
        
        # Combine recommendations
        full_recommendation = f"""
# 🤖 AI-Powered Treatment Recommendation
*Generated using {ai_source} with evidence-based medical guidelines*

---

{ai_recommendation}

{clinical_guidelines}

---

## ⚠️ IMPORTANT DISCLAIMER
This recommendation is generated by AI based on clinical data and current medical guidelines. 
It is intended for educational purposes and to assist healthcare providers in clinical decision-making. 
**Always consult with a qualified healthcare professional for proper diagnosis and treatment.**

**This recommendation should not replace professional medical advice, diagnosis, or treatment.**
        """
        
        return full_recommendation
        
    except Exception as e:
        # Fallback recommendation based on risk level
        fallback = engine.create_fallback_recommendation(patient_data, risk_probability, risk_level)
        return f"""
# 🏥 Clinical Guideline-Based Recommendation
*AI service temporarily unavailable - showing evidence-based clinical guidelines*

{fallback}

---

## ⚠️ IMPORTANT DISCLAIMER
This recommendation is based on standard clinical guidelines for H. pylori management.
**Always consult with a qualified healthcare professional for proper diagnosis and treatment.**
        """

def create_fallback_recommendation(patient_data: Dict[str, Any], 
                                 risk_probability: float, 
                                 risk_level: str) -> str:
    """
    Create a fallback recommendation when AI services are unavailable
    """
    
    engine = AIRecommendationEngine()
    
    risk_guidelines = engine.medical_guidelines.get(risk_level.lower(), {})
    
    recommendation = f"""
## Risk Assessment
- **H. Pylori Infection Probability**: {risk_probability:.1%}
- **Risk Level**: {risk_level.upper()}
- **Clinical Priority**: {risk_guidelines.get('urgency', 'Standard care')}

## Recommended Actions

### Diagnostic Testing
{chr(10).join(['- ' + test for test in risk_guidelines.get('testing', ['Standard H. pylori testing'])])}

### Treatment Approach
- **Strategy**: {risk_guidelines.get('treatment_approach', 'Follow standard guidelines')}
- **Follow-up**: {risk_guidelines.get('followup', 'As clinically indicated')}

### Key Risk Factors Identified
    """
    
    # Add risk factor analysis
    risk_factors = []
    
    if patient_data.get('Gastritis_History', 0) == 1:
        risk_factors.append("Previous gastritis history")
    
    if patient_data.get('Ulcer_History', 0) == 1:
        risk_factors.append("Previous ulcer disease")
    
    if patient_data.get('Family_Pylori_History', 0) == 1:
        risk_factors.append("Family history of H. pylori")
    
    if patient_data.get('Smoking', 0) > 0:
        risk_factors.append("Smoking history")
    
    if patient_data.get('Pickled_Food', 0) > 2:
        risk_factors.append("High pickled food consumption")
    
    if risk_factors:
        recommendation += "\n" + "\n".join(['- ' + factor for factor in risk_factors])
    else:
        recommendation += "\n- No major risk factors identified"
    
    recommendation += f"""

### Lifestyle Recommendations
- Improve hand hygiene practices
- Reduce consumption of pickled and processed foods
- Consider smoking cessation if applicable
- Maintain good nutritional status
- Follow proper food safety practices

### Follow-up Care
- {risk_guidelines.get('followup', 'Schedule appropriate follow-up based on clinical judgment')}
- Monitor symptoms and report any worsening
- Ensure treatment compliance if therapy is initiated
    """
    
    return recommendation

# Attach method to class
AIRecommendationEngine.create_fallback_recommendation = create_fallback_recommendation

if __name__ == "__main__":
    # Test the recommendation system
    test_patient = {
        'Age': 45,
        'Sex': 1,
        'BMI': 26.5,
        'Gastritis_History': 1,
        'Ulcer_History': 0,
        'Family_Pylori_History': 1,
        'Smoking': 1,
        'Alcohol': 2,
        'Handwashing': 2,
        'Pickled_Food': 2
    }
    
    try:
        recommendation = get_ai_treatment_recommendation(
            test_patient, 0.75, "High"
        )
        print("AI Recommendation Generated Successfully!")
        print(recommendation[:500] + "...")
    except Exception as e:
        print(f"Error: {e}")
