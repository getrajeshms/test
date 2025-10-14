import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    """
    Comprehensive ML Pipeline for H. Pylori infection prediction
    """
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.model_results = {}
        self.feature_importances = {}
        self.roc_curves = {}
        self.confusion_matrices = {}
        
    def create_preprocessor(self, X):
        """
        Create preprocessing pipeline for features
        """
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Handle missing values (-1) in endoscopy features
        endoscopy_features = ['Nodularity', 'Gastric_Redness']
        
        # Numeric preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def engineer_features(self, df):
        """
        Create derived features and feature interactions
        """
        df_engineered = df.copy()
        
        # Age groups
        df_engineered['Age_Group'] = pd.cut(
            df_engineered['Age'], 
            bins=[0, 30, 45, 60, 100], 
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # BMI categories
        bmi_conditions = [
            (df_engineered['BMI'] < 18.5),
            (df_engineered['BMI'] >= 18.5) & (df_engineered['BMI'] < 25),
            (df_engineered['BMI'] >= 25) & (df_engineered['BMI'] < 30),
            (df_engineered['BMI'] >= 30)
        ]
        bmi_choices = [0, 1, 2, 3]  # Underweight, Normal, Overweight, Obese
        df_engineered['BMI_Category'] = np.select(bmi_conditions, bmi_choices)
        
        # Lifestyle risk score
        df_engineered['Lifestyle_Risk'] = (
            df_engineered['Smoking'] * 0.3 +
            df_engineered['Alcohol'] * 0.2 +
            df_engineered['Pickled_Food'] * 0.2 +
            (3 - df_engineered['Handwashing']) * 0.3  # Reverse handwashing (less = more risk)
        )
        
        # Family risk score
        df_engineered['Family_Risk'] = (
            df_engineered['Family_Pylori_History'] * 0.6 +
            df_engineered['Family_Gastritis_History'] * 0.4
        )
        
        # Clinical history score
        df_engineered['Clinical_History_Score'] = (
            df_engineered['Gastritis_History'] * 0.5 +
            df_engineered['Ulcer_History'] * 0.5
        )
        
        # Blood parameter ratios (common clinical indicators)
        df_engineered['Neutrophil_Lymphocyte_Ratio'] = (
            df_engineered['Neutrophil_Count'] / (df_engineered['Lymphocyte_Count'] + 0.1)
        )
        
        # Socioeconomic indicator
        df_engineered['Socioeconomic_Score'] = (
            df_engineered['Education'] * 0.4 +
            (3 - df_engineered['Residence']) * 0.3 +  # Urban = higher score
            df_engineered['Water_Source'] * 0.3
        )
        
        # Interaction terms
        df_engineered['Age_BMI'] = df_engineered['Age'] * df_engineered['BMI'] / 100
        df_engineered['Family_Size_Sharing'] = (
            df_engineered['Family_Size'] * df_engineered['Tableware_Sharing'] / 10
        )
        
        # Handle missing endoscopy data
        df_engineered['Has_Endoscopy'] = (
            (df_engineered['Nodularity'] != -1) & (df_engineered['Gastric_Redness'] != -1)
        ).astype(int)
        
        # Convert missing endoscopy values to 0 for modeling
        df_engineered['Nodularity'] = df_engineered['Nodularity'].replace(-1, 0)
        df_engineered['Gastric_Redness'] = df_engineered['Gastric_Redness'].replace(-1, 0)
        
        return df_engineered
    
    def setup_models(self):
        """
        Initialize all models to be evaluated
        """
        self.models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=1,
                max_depth=6,
                learning_rate=0.1
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                class_weight='balanced',
                kernel='rbf'
            )
        }
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Evaluate a single model and return metrics
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return metrics, (fpr, tpr, metrics['roc_auc']), cm
    
    def train_models(self, data):
        """
        Train and evaluate all models
        """
        # Feature engineering
        print("Performing feature engineering...")
        data_engineered = self.engineer_features(data)
        
        # Separate features and target
        X = data_engineered.drop('H_Pylori_Infection', axis=1)
        y = data_engineered['H_Pylori_Infection']
        
        print(f"Dataset shape after feature engineering: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Create preprocessor
        self.create_preprocessor(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocess data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
        
        print(f"After SMOTE - Training set shape: {X_train_balanced.shape}")
        print(f"After SMOTE - Target distribution: {np.bincount(y_train_balanced)}")
        
        # Setup models
        self.setup_models()
        
        # Train and evaluate each model
        best_score = 0
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Evaluate model
                metrics, roc_data, cm = self.evaluate_model(
                    model, X_train_balanced, X_test_processed, y_train_balanced, y_test
                )
                
                # Store results
                self.model_results[name] = metrics
                self.roc_curves[name] = roc_data
                self.confusion_matrices[name] = cm
                
                # Check if this is the best model
                if metrics['roc_auc'] > best_score:
                    best_score = metrics['roc_auc']
                    self.best_model = model
                    self.best_model_name = name
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importances[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    self.feature_importances[name] = np.abs(model.coef_[0])
                
                print(f"{name} - ROC AUC: {metrics['roc_auc']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        print(f"\nBest model: {self.best_model_name} with ROC AUC: {best_score:.3f}")
        
        # Perform cross-validation on best model
        print("\nPerforming cross-validation on best model...")
        cv_scores = cross_val_score(
            self.best_model, X_train_balanced, y_train_balanced, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        print(f"Cross-validation ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.model_results
    
    def predict(self, X):
        """
        Make predictions using the best model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        # Feature engineering
        X_engineered = self.engineer_features(X)
        
        # Remove target if present
        if 'H_Pylori_Infection' in X_engineered.columns:
            X_engineered = X_engineered.drop('H_Pylori_Infection', axis=1)
        
        # Preprocess
        X_processed = self.preprocessor.transform(X_engineered)
        
        return self.best_model.predict(X_processed)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities using the best model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        # Feature engineering
        X_engineered = self.engineer_features(X)
        
        # Remove target if present
        if 'H_Pylori_Infection' in X_engineered.columns:
            X_engineered = X_engineered.drop('H_Pylori_Infection', axis=1)
        
        # Preprocess
        X_processed = self.preprocessor.transform(X_engineered)
        
        return self.best_model.predict_proba(X_processed)
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from the best model
        """
        if self.best_model_name not in self.feature_importances:
            return None
        
        feature_names = self.preprocessor.get_feature_names_out()
        importances = self.feature_importances[self.best_model_name]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, model_path, preprocessor_path):
        """
        Save the trained model and preprocessor
        """
        if self.best_model is None:
            raise ValueError("No model to save!")
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        
        # Save additional metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'model_results': self.model_results,
            'feature_importances': self.feature_importances,
            'roc_curves': self.roc_curves,
            'confusion_matrices': self.confusion_matrices
        }
        joblib.dump(metadata, model_path.replace('.joblib', '_metadata.joblib'))
    
    def load_model(self, model_path, preprocessor_path):
        """
        Load a trained model and preprocessor
        """
        self.best_model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load metadata if available
        try:
            metadata_path = model_path.replace('.joblib', '_metadata.joblib')
            metadata = joblib.load(metadata_path)
            self.best_model_name = metadata.get('best_model_name', 'Unknown')
            self.model_results = metadata.get('model_results', {})
            self.feature_importances = metadata.get('feature_importances', {})
            self.roc_curves = metadata.get('roc_curves', {})
            self.confusion_matrices = metadata.get('confusion_matrices', {})
            print(f"✓ Loaded model metadata: {self.best_model_name}")
            print(f"✓ Model results available: {len(self.model_results)} models")
            print(f"✓ ROC curves available: {len(self.roc_curves)} models")
        except Exception as e:
            print(f"Warning: Could not load model metadata: {e}")
            self.best_model_name = 'Unknown'
            self.model_results = {}
            self.feature_importances = {}
    
    def generate_classification_report(self, X_test, y_test):
        """
        Generate detailed classification report
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        predictions = self.predict(X_test)
        report = classification_report(y_test, predictions, 
                                     target_names=['No Infection', 'Infection'])
        return report

if __name__ == "__main__":
    # Test the ML pipeline
    from data_generator import generate_synthetic_data
    
    print("Testing ML Pipeline...")
    
    # Generate test data
    data = generate_synthetic_data(1000)
    
    # Initialize and train pipeline
    pipeline = MLPipeline()
    results = pipeline.train_models(data)
    
    print("\nModel Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: ROC AUC = {metrics['roc_auc']:.3f}")
    
    # Test prediction
    test_patient = data.iloc[[0]].drop('H_Pylori_Infection', axis=1)
    prediction = pipeline.predict_proba(test_patient)
    print(f"\nTest prediction probability: {prediction[0][1]:.3f}")
