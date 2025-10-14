import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import warnings
warnings.filterwarnings('ignore')

class ModelUtils:
    """
    Utility class for model visualization and interpretation
    """
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """
        Plot confusion matrix using plotly
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = ['No Infection', 'Infection']
            
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create text annotations
        text = []
        for i in range(len(labels)):
            text_row = []
            for j in range(len(labels)):
                text_row.append(f'{cm[i,j]}<br>({cm_percent[i,j]:.1f}%)')
            text.append(text_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=500,
            height=400
        )
        
        return fig
    
    def plot_roc_curves(self, models_data, title="ROC Curves Comparison"):
        """
        Plot ROC curves for multiple models
        
        Args:
            models_data: Dict with model names as keys and (fpr, tpr, auc_score) as values
        """
        fig = go.Figure()
        
        for i, (model_name, (fpr, tpr, auc_score)) in enumerate(models_data.items()):
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {auc_score:.3f})',
                line=dict(color=self.colors[i % len(self.colors)], width=2)
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray', width=1)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=700,
            height=500,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_scores, title="Precision-Recall Curve"):
        """
        Plot precision-recall curve
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = auc(recall, precision)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'Precision-Recall (AP = {avg_precision:.3f})',
            line=dict(color=self.colors[0], width=2)
        ))
        
        # Add baseline
        baseline = y_true.mean()
        fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                     annotation_text=f"Baseline (AP = {baseline:.3f})")
        
        fig.update_layout(
            title=title,
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=400
        )
        
        return fig
    
    def plot_feature_importance(self, feature_names, importances, top_n=20, 
                               title="Feature Importance"):
        """
        Plot feature importance
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=title,
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=800,
            height=max(400, top_n * 25),
            yaxis_title='Features',
            xaxis_title='Importance Score'
        )
        
        return fig
    
    def plot_learning_curves(self, estimator, X, y, cv=5, title="Learning Curves"):
        """
        Plot learning curves to assess model performance vs training size
        """
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color=self.colors[0]),
            error_y=dict(
                type='data',
                array=train_std,
                visible=True
            )
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=self.colors[1]),
            error_y=dict(
                type='data',
                array=val_std,
                visible=True
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Training Set Size',
            yaxis_title='ROC AUC Score',
            width=700,
            height=400
        )
        
        return fig
    
    def plot_prediction_distribution(self, y_true, y_pred_proba, title="Prediction Distribution"):
        """
        Plot distribution of prediction probabilities by true class
        """
        df = pd.DataFrame({
            'probability': y_pred_proba,
            'true_class': ['Infection' if x == 1 else 'No Infection' for x in y_true]
        })
        
        fig = px.histogram(
            df,
            x='probability',
            color='true_class',
            nbins=50,
            title=title,
            opacity=0.7,
            barmode='overlay'
        )
        
        fig.update_layout(
            xaxis_title='Predicted Probability of Infection',
            yaxis_title='Count',
            width=700,
            height=400
        )
        
        return fig
    
    def create_shap_summary_plot(self, model, X_sample, feature_names=None):
        """
        Create SHAP summary plot for model interpretation
        """
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # If binary classification, use positive class values
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Create summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.title("SHAP Feature Importance Summary")
            plt.tight_layout()
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Could not create SHAP plot: {e}")
            return None
    
    def plot_calibration_curve(self, y_true, y_prob, n_bins=10, title="Calibration Curve"):
        """
        Plot calibration curve to assess probability calibration
        """
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins
        )
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        # Calibration curve
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color=self.colors[0], width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=400
        )
        
        return fig
    
    def create_model_comparison_dashboard(self, model_results):
        """
        Create comprehensive model comparison dashboard
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        models = list(model_results.keys())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Overall Comparison'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "radar"}]]
        )
        
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        
        # Individual metric plots
        for i, metric in enumerate(metrics):
            if i < 5:
                row, col = positions[i]
                values = [model_results[model][metric] for model in models]
                
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=values,
                        name=metric.title(),
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # Radar chart for overall comparison
        for i, model in enumerate(models):
            values = [model_results[model][metric] for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=model,
                    line_color=self.colors[i % len(self.colors)]
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title="Model Performance Comparison Dashboard",
            height=800,
            width=1200,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            )
        )
        
        return fig
    
    def plot_risk_stratification(self, y_true, y_pred_proba, title="Risk Stratification Analysis"):
        """
        Analyze model performance across different risk thresholds
        """
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        sensitivity_scores = []
        specificity_scores = []
        ppv_scores = []
        npv_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            sensitivity_scores.append(sensitivity)
            specificity_scores.append(specificity)
            ppv_scores.append(ppv)
            npv_scores.append(npv)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=sensitivity_scores,
            mode='lines+markers',
            name='Sensitivity',
            line=dict(color=self.colors[0])
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=specificity_scores,
            mode='lines+markers',
            name='Specificity',
            line=dict(color=self.colors[1])
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=ppv_scores,
            mode='lines+markers',
            name='PPV',
            line=dict(color=self.colors[2])
        ))
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=npv_scores,
            mode='lines+markers',
            name='NPV',
            line=dict(color=self.colors[3])
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Prediction Threshold',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            width=700,
            height=400
        )
        
        return fig
    
    def calculate_clinical_metrics(self, y_true, y_pred_proba, threshold=0.5):
        """
        Calculate clinical performance metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'prevalence': (tp + fn) / (tp + tn + fp + fn),
            'lr_positive': (tp / (tp + fn)) / (fp / (tn + fp)) if (fp / (tn + fp)) > 0 else float('inf'),
            'lr_negative': (fn / (tp + fn)) / (tn / (tn + fp)) if (tn / (tn + fp)) > 0 else float('inf')
        }
        
        return metrics

if __name__ == "__main__":
    # Test the utility functions
    print("Model utilities loaded successfully!")
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_pred_proba = np.random.beta(2, 5, n_samples)
    y_pred_proba[y_true == 1] += np.random.normal(0, 0.2, (y_true == 1).sum())
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    utils = ModelUtils()
    
    # Test clinical metrics
    metrics = utils.calculate_clinical_metrics(y_true, y_pred_proba)
    print("Clinical metrics calculated successfully!")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
