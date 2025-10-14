import pandas as pd
import numpy as np
from scipy import stats
import random

def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generate synthetic H. pylori patient data based on medical literature and realistic distributions
    
    Parameters:
    n_samples (int): Number of synthetic patient records to generate
    random_state (int): Random seed for reproducibility
    
    Returns:
    pandas.DataFrame: Synthetic patient dataset with all relevant features
    """
    
    np.random.seed(random_state)
    random.seed(random_state)
    
    data = {}
    
    # Demographics
    data['Age'] = np.random.normal(45, 15, n_samples).astype(int)
    data['Age'] = np.clip(data['Age'], 18, 75)  # Age range 18-75
    
    data['Sex'] = np.random.binomial(1, 0.52, n_samples)  # 52% male
    
    # Residence: 0=Urban, 1=County, 2=Suburban, 3=Village
    residence_probs = [0.35, 0.25, 0.25, 0.15]
    data['Residence'] = np.random.choice(4, n_samples, p=residence_probs)
    
    # Education: 0=Primary, 1=Secondary, 2=College, 3=Bachelor, 4=Postgraduate
    education_probs = [0.15, 0.25, 0.25, 0.25, 0.10]
    data['Education'] = np.random.choice(5, n_samples, p=education_probs)
    
    # Marital Status: 0=Unmarried, 1=Married
    data['Marital_Status'] = np.random.binomial(1, 0.65, n_samples)  # 65% married
    
    # Family Size (1-8 members)
    data['Family_Size'] = np.random.poisson(3.2, n_samples) + 1
    data['Family_Size'] = np.clip(data['Family_Size'], 1, 8)
    
    # Clinical History and Symptoms
    data['BMI'] = np.random.normal(25.5, 4.2, n_samples)
    data['BMI'] = np.clip(data['BMI'], 16, 45)
    
    # Non-atrophic gastritis history (higher in H. pylori positive)
    base_gastritis_rate = 0.20
    data['Gastritis_History'] = np.random.binomial(1, base_gastritis_rate, n_samples)
    
    # Ulcer/Peptic ulcer history
    base_ulcer_rate = 0.15
    data['Ulcer_History'] = np.random.binomial(1, base_ulcer_rate, n_samples)
    
    # Laboratory Values
    data['Albumin'] = np.random.normal(42, 4.5, n_samples)
    data['Albumin'] = np.clip(data['Albumin'], 28, 52)
    
    data['WBC_Count'] = np.random.normal(7.2, 1.8, n_samples)
    data['WBC_Count'] = np.clip(data['WBC_Count'], 3.5, 12)
    
    data['Lymphocyte_Count'] = np.random.normal(2.1, 0.6, n_samples)
    data['Lymphocyte_Count'] = np.clip(data['Lymphocyte_Count'], 0.8, 4.5)
    
    data['Neutrophil_Count'] = np.random.normal(4.2, 1.2, n_samples)
    data['Neutrophil_Count'] = np.clip(data['Neutrophil_Count'], 1.5, 8.5)
    
    data['RBC_Count'] = np.random.normal(4.5, 0.4, n_samples)
    data['RBC_Count'] = np.clip(data['RBC_Count'], 3.2, 5.8)
    
    # Hemoglobin (slightly lower in H. pylori positive due to potential iron deficiency)
    data['Hemoglobin'] = np.random.normal(135, 15, n_samples)
    data['Hemoglobin'] = np.clip(data['Hemoglobin'], 90, 170)
    
    # Lifestyle Factors
    # Smoking: 0=None, 1=1-5/day, 2=6-10/day, 3=>10/day
    smoking_probs = [0.65, 0.20, 0.10, 0.05]
    data['Smoking'] = np.random.choice(4, n_samples, p=smoking_probs)
    
    # Alcohol: 0=None, 1=Monthly, 2=Weekly, 3=3+/Weekly
    alcohol_probs = [0.45, 0.25, 0.20, 0.10]
    data['Alcohol'] = np.random.choice(4, n_samples, p=alcohol_probs)
    
    # Water source: 0=Tap, 1=Purified, 2=Mineral, 3=Other
    water_probs = [0.40, 0.35, 0.20, 0.05]
    data['Water_Source'] = np.random.choice(4, n_samples, p=water_probs)
    
    # Pickled food consumption: 0=Rare, 1=Now&then, 2=Frequent, 3=Daily
    pickled_probs = [0.30, 0.40, 0.20, 0.10]
    data['Pickled_Food'] = np.random.choice(4, n_samples, p=pickled_probs)
    
    # Handwashing frequency: 0=Rarely, 1=Now&then, 2=Frequent, 3=Daily
    handwashing_probs = [0.05, 0.15, 0.35, 0.45]
    data['Handwashing'] = np.random.choice(4, n_samples, p=handwashing_probs)
    
    # Tableware sharing frequency: 0=Rare, 1=Now&then, 2=Frequent, 3=Daily
    sharing_probs = [0.25, 0.35, 0.25, 0.15]
    data['Tableware_Sharing'] = np.random.choice(4, n_samples, p=sharing_probs)
    
    # Family History
    data['Family_Pylori_History'] = np.random.binomial(1, 0.18, n_samples)
    data['Family_Gastritis_History'] = np.random.binomial(1, 0.22, n_samples)
    
    # Endoscopy/Imaging Features (Optional - some missing values)
    # Nodularity: gastric nodularity on endoscopy
    nodularity_available = np.random.binomial(1, 0.7, n_samples)  # 70% have endoscopy data
    data['Nodularity'] = np.where(
        nodularity_available, 
        np.random.binomial(1, 0.25, n_samples), 
        -1  # Missing value indicator
    )
    
    # Gastric mucosal redness
    redness_available = np.random.binomial(1, 0.7, n_samples)
    data['Gastric_Redness'] = np.where(
        redness_available,
        np.random.binomial(1, 0.35, n_samples),
        -1  # Missing value indicator
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Generate H. Pylori infection status based on realistic risk factors
    # Create a risk score based on known risk factors
    risk_score = (
        0.05 * (df['Age'] - 40) +  # Age factor
        0.1 * df['Sex'] +  # Male slightly higher risk
        0.15 * (df['Residence'] > 1) +  # Rural residence
        -0.1 * df['Education'] +  # Higher education protective
        0.2 * df['Family_Size'] / 8 +  # Crowded living
        0.3 * df['Gastritis_History'] +  # Previous gastritis
        0.25 * df['Ulcer_History'] +  # Previous ulcer
        -0.05 * (df['Albumin'] - 40) / 10 +  # Nutritional status
        0.1 * df['Smoking'] / 3 +  # Smoking
        -0.15 * df['Handwashing'] / 3 +  # Hygiene
        0.1 * df['Tableware_Sharing'] / 3 +  # Sharing utensils
        0.2 * df['Family_Pylori_History'] +  # Family history
        0.1 * df['Family_Gastritis_History'] +  # Family gastritis
        0.1 * df['Pickled_Food'] / 3 +  # Dietary factors
        0.05 * (df['Water_Source'] == 3) +  # Poor water quality
        np.random.normal(0, 0.3, n_samples)  # Random variation
    )
    
    # Convert risk score to probability using sigmoid function
    probability = 1 / (1 + np.exp(-risk_score))
    
    # Generate binary outcome with realistic prevalence (around 30-40%)
    # Adjust baseline to get desired prevalence
    adjusted_prob = (probability - probability.mean() + 0.35)
    adjusted_prob = np.clip(adjusted_prob, 0.05, 0.95)
    
    df['H_Pylori_Infection'] = np.random.binomial(1, adjusted_prob, n_samples)
    
    # Adjust some features based on infection status to make relationships more realistic
    infection_mask = df['H_Pylori_Infection'] == 1
    
    # H. pylori positive patients more likely to have gastritis/ulcer history
    df.loc[infection_mask & (np.random.random(n_samples) < 0.3), 'Gastritis_History'] = 1
    df.loc[infection_mask & (np.random.random(n_samples) < 0.2), 'Ulcer_History'] = 1
    
    # Slightly lower hemoglobin in infected patients (iron deficiency)
    df.loc[infection_mask, 'Hemoglobin'] -= np.random.normal(5, 3, infection_mask.sum())
    df['Hemoglobin'] = np.clip(df['Hemoglobin'], 90, 170)
    
    # Higher WBC in some infected patients (inflammatory response)
    inflammatory_response = infection_mask & (np.random.random(n_samples) < 0.3)
    df.loc[inflammatory_response, 'WBC_Count'] += np.random.normal(1.5, 0.8, inflammatory_response.sum())
    df['WBC_Count'] = np.clip(df['WBC_Count'], 3.5, 12)
    
    # Endoscopy findings more common in infected patients
    df.loc[infection_mask & (df['Nodularity'] != -1) & (np.random.random(n_samples) < 0.4), 'Nodularity'] = 1
    df.loc[infection_mask & (df['Gastric_Redness'] != -1) & (np.random.random(n_samples) < 0.5), 'Gastric_Redness'] = 1
    
    # Round numeric values appropriately
    df['Age'] = df['Age'].astype(int)
    df['Family_Size'] = df['Family_Size'].astype(int)
    df['BMI'] = df['BMI'].round(1)
    df['Albumin'] = df['Albumin'].round(1)
    df['WBC_Count'] = df['WBC_Count'].round(1)
    df['Lymphocyte_Count'] = df['Lymphocyte_Count'].round(1)
    df['Neutrophil_Count'] = df['Neutrophil_Count'].round(1)
    df['RBC_Count'] = df['RBC_Count'].round(1)
    df['Hemoglobin'] = df['Hemoglobin'].round(0).astype(int)
    
    # Add feature labels for interpretability
    feature_labels = {
        'Age': 'Patient age in years',
        'Sex': 'Biological sex (0=Female, 1=Male)',
        'Residence': 'Type of residence (0=Urban, 1=County, 2=Suburban, 3=Village)',
        'Education': 'Education level (0=Primary, 1=Secondary, 2=College, 3=Bachelor, 4=Postgraduate)',
        'Marital_Status': 'Marital status (0=Unmarried, 1=Married)',
        'Family_Size': 'Number of household members',
        'BMI': 'Body Mass Index (kg/m²)',
        'Gastritis_History': 'History of non-atrophic gastritis (0=No, 1=Yes)',
        'Ulcer_History': 'History of ulcer/peptic ulcer (0=No, 1=Yes)',
        'Albumin': 'Serum albumin concentration (g/L)',
        'WBC_Count': 'White blood cell count (10⁹/L)',
        'Lymphocyte_Count': 'Absolute lymphocyte count (10⁹/L)',
        'Neutrophil_Count': 'Absolute neutrophil count (10⁹/L)',
        'RBC_Count': 'Red blood cell count (10¹²/L)',
        'Hemoglobin': 'Hemoglobin level (g/L)',
        'Smoking': 'Smoking status (0=None, 1=1-5/day, 2=6-10/day, 3=>10/day)',
        'Alcohol': 'Alcohol consumption (0=None, 1=Monthly, 2=Weekly, 3=3+/Weekly)',
        'Water_Source': 'Drinking water source (0=Tap, 1=Purified, 2=Mineral, 3=Other)',
        'Pickled_Food': 'Pickled food consumption (0=Rare, 1=Now&then, 2=Frequent, 3=Daily)',
        'Handwashing': 'Handwashing frequency (0=Rarely, 1=Now&then, 2=Frequent, 3=Daily)',
        'Tableware_Sharing': 'Frequency of sharing tableware (0=Rare, 1=Now&then, 2=Frequent, 3=Daily)',
        'Family_Pylori_History': 'Family history of H. pylori (0=No, 1=Yes)',
        'Family_Gastritis_History': 'Family history of gastritis (0=No, 1=Yes)',
        'Nodularity': 'Gastric nodularity on endoscopy (0=No, 1=Yes, -1=Not available)',
        'Gastric_Redness': 'Gastric mucosal redness (0=No, 1=Yes, -1=Not available)',
        'H_Pylori_Infection': 'H. pylori infection status (0=Negative, 1=Positive)'
    }
    
    # Add metadata
    df.attrs['feature_labels'] = feature_labels
    df.attrs['n_samples'] = n_samples
    df.attrs['prevalence'] = df['H_Pylori_Infection'].mean()
    
    return df

def get_data_description():
    """
    Returns detailed description of the synthetic dataset structure and ranges
    """
    description = {
        'demographics': {
            'Age': {'range': '18-75 years', 'distribution': 'Normal(45, 15)', 'type': 'continuous'},
            'Sex': {'values': '0=Female, 1=Male', 'prevalence': '52% male', 'type': 'binary'},
            'Residence': {'values': '0=Urban, 1=County, 2=Suburban, 3=Village', 'distribution': '[35%, 25%, 25%, 15%]', 'type': 'categorical'},
            'Education': {'values': '0=Primary, 1=Secondary, 2=College, 3=Bachelor, 4=Postgraduate', 'distribution': '[15%, 25%, 25%, 25%, 10%]', 'type': 'ordinal'},
            'Marital_Status': {'values': '0=Unmarried, 1=Married', 'prevalence': '65% married', 'type': 'binary'},
            'Family_Size': {'range': '1-8 members', 'distribution': 'Poisson(3.2)', 'type': 'count'}
        },
        'clinical': {
            'BMI': {'range': '16-45 kg/m²', 'distribution': 'Normal(25.5, 4.2)', 'type': 'continuous'},
            'Gastritis_History': {'values': '0=No, 1=Yes', 'prevalence': '~20%', 'type': 'binary'},
            'Ulcer_History': {'values': '0=No, 1=Yes', 'prevalence': '~15%', 'type': 'binary'}
        },
        'laboratory': {
            'Albumin': {'range': '28-52 g/L', 'distribution': 'Normal(42, 4.5)', 'type': 'continuous'},
            'WBC_Count': {'range': '3.5-12 ×10⁹/L', 'distribution': 'Normal(7.2, 1.8)', 'type': 'continuous'},
            'Lymphocyte_Count': {'range': '0.8-4.5 ×10⁹/L', 'distribution': 'Normal(2.1, 0.6)', 'type': 'continuous'},
            'Neutrophil_Count': {'range': '1.5-8.5 ×10⁹/L', 'distribution': 'Normal(4.2, 1.2)', 'type': 'continuous'},
            'RBC_Count': {'range': '3.2-5.8 ×10¹²/L', 'distribution': 'Normal(4.5, 0.4)', 'type': 'continuous'},
            'Hemoglobin': {'range': '90-170 g/L', 'distribution': 'Normal(135, 15)', 'type': 'continuous'}
        },
        'lifestyle': {
            'Smoking': {'values': '0=None, 1=1-5/day, 2=6-10/day, 3=>10/day', 'distribution': '[65%, 20%, 10%, 5%]', 'type': 'ordinal'},
            'Alcohol': {'values': '0=None, 1=Monthly, 2=Weekly, 3=3+/Weekly', 'distribution': '[45%, 25%, 20%, 10%]', 'type': 'ordinal'},
            'Water_Source': {'values': '0=Tap, 1=Purified, 2=Mineral, 3=Other', 'distribution': '[40%, 35%, 20%, 5%]', 'type': 'categorical'},
            'Pickled_Food': {'values': '0=Rare, 1=Now&then, 2=Frequent, 3=Daily', 'distribution': '[30%, 40%, 20%, 10%]', 'type': 'ordinal'},
            'Handwashing': {'values': '0=Rarely, 1=Now&then, 2=Frequent, 3=Daily', 'distribution': '[5%, 15%, 35%, 45%]', 'type': 'ordinal'},
            'Tableware_Sharing': {'values': '0=Rare, 1=Now&then, 2=Frequent, 3=Daily', 'distribution': '[25%, 35%, 25%, 15%]', 'type': 'ordinal'}
        },
        'family_history': {
            'Family_Pylori_History': {'values': '0=No, 1=Yes', 'prevalence': '~18%', 'type': 'binary'},
            'Family_Gastritis_History': {'values': '0=No, 1=Yes', 'prevalence': '~22%', 'type': 'binary'}
        },
        'endoscopy': {
            'Nodularity': {'values': '0=No, 1=Yes, -1=Not available', 'availability': '70%', 'type': 'binary_with_missing'},
            'Gastric_Redness': {'values': '0=No, 1=Yes, -1=Not available', 'availability': '70%', 'type': 'binary_with_missing'}
        },
        'target': {
            'H_Pylori_Infection': {'values': '0=Negative, 1=Positive', 'prevalence': '~35%', 'type': 'binary'}
        }
    }
    
    return description

if __name__ == "__main__":
    # Test data generation
    print("Generating synthetic H. pylori dataset...")
    df = generate_synthetic_data(1000)
    
    print(f"Generated dataset shape: {df.shape}")
    print(f"H. pylori prevalence: {df['H_Pylori_Infection'].mean():.3f}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset summary:")
    print(df.describe())
