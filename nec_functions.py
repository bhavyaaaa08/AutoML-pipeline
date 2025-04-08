import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix,
                            precision_recall_curve, roc_curve, auc, 
                            mean_squared_error, r2_score, mean_absolute_error)
import optuna
from IPython.display import display
import io
import base64
import logging
import warnings
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def sanitize_feature_names(feature_names):
    """Clean up feature names to avoid issues with SHAP."""
    sanitized = []
    for name in feature_names:
        name = str(name).replace('<', '').replace('>', '').replace('=', '')
        name = name.replace('[', '').replace(']', '').replace(':', '_')
        sanitized.append(name)
    return sanitized

def preprocess_data(df, target_col, test_size=0.2, random_state=42):
    """Process data for modeling with enhanced error handling and security."""
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        df = df.copy()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if df.isna().any().any():
            logger.warning(f"Dataset contains missing values. These will be imputed.")
        
        initial_rows = df.shape[0]
        df.drop_duplicates(inplace=True)
        if df.shape[0] < initial_rows:
            logger.info(f"Removed {initial_rows - df.shape[0]} duplicate rows")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        label_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category' or y.dtype == 'bool':
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            class_names = label_encoder.classes_
            logger.info(f"Encoded target classes: {class_names}")
        
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        transformers = []
        if numerical_cols:
            transformers.append(('num', numerical_pipeline, numerical_cols))
        if categorical_cols:
            transformers.append(('cat', categorical_pipeline, categorical_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) < 10 else None
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = sanitize_feature_names(feature_names)
        except:
            feature_names = []
            for name, _, cols in transformers:
                for col in cols:
                    if name == 'num':
                        feature_names.append(f"{name}_{col}")
                    else:
                        unique_vals = df[col].nunique()
                        for i in range(unique_vals):
                            feature_names.append(f"{name}_{col}_{i}")
        
        logger.info(f"Data preprocessing complete. X_train shape: {X_train_processed.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_processed': X_train_processed,
            'X_test_processed': X_test_processed,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'label_encoder': label_encoder,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols
        }
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def determine_task_type(y):
    """Determine if the problem is classification or regression."""
    unique_values = np.unique(y)
    if len(unique_values) < 10 or isinstance(y, (pd.Categorical, pd.Series)) and pd.api.types.is_categorical_dtype(y):
        return 'classification'
    else:
        return 'regression'

def objective(trial, X_train, y_train, X_test, y_test, task_type):
    """Optimization objective function for Optuna."""
    try:
        if task_type == 'classification':
            model_type = trial.suggest_categorical('model', ['logistic', 'random_forest'])
            if model_type == 'logistic':
                C = trial.suggest_float('logreg_C', 1e-4, 1e2, log=True)
                solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
                model = LogisticRegression(C=C, solver=solver, penalty=penalty, max_iter=1000, random_state=42)
            else:
                n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
                max_depth = trial.suggest_int('rf_max_depth', 2, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )

            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return scores.mean()
            
        else:
            model_type = trial.suggest_categorical('model', ['linear', 'random_forest'])
            if model_type == 'linear':
                model = LinearRegression()
            else:
                n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
                max_depth = trial.suggest_int('rf_max_depth', 2, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                model = RandomForestRegressor(
                    n_estimators=n_estimators, 
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )

            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            return scores.mean()
    except Exception as e:
        logger.error(f"Error in objective function: {str(e)}")
        return float('-inf')

def generate_confusion_matrix_plot(y_test, y_pred, class_names=None):
    """Generate a confusion matrix plot."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names if class_names is not None else 'auto',
                yticklabels=class_names if class_names is not None else 'auto')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_roc_curve_plot(y_test, y_prob):
    """Generate ROC curve plot for classification."""
    plt.figure(figsize=(8, 6))
    
    if len(y_prob.shape) > 1 and y_prob.shape[1] > 2:
        n_classes = y_prob.shape[1]
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
    else:
        if len(y_prob.shape) > 1:
            probs = y_prob[:, 1]
        else:
            probs = y_prob
            
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_residual_plot(y_test, y_pred):
    """Generate residual plot for regression."""
    plt.figure(figsize=(10, 6))
    
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_actual_vs_predicted_plot(y_test, y_pred):
    """Generate actual vs predicted values plot for regression."""
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_test, y_pred, alpha=0.6)
    
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_feature_importance_plot(model, feature_names, top_n=20):
    """Generate feature importance plot."""
    plt.figure(figsize=(12, 8))
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            
            plt.title('Feature Importance')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices])
            plt.xlabel('Relative Importance')
            
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                importances = np.mean(np.abs(coefs), axis=0)
            else:
                importances = np.abs(coefs)
                
            indices = np.argsort(importances)[-top_n:]
            plt.title('Feature Importance (Coefficient Magnitude)')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices])
            plt.xlabel('Absolute Coefficient Magnitude')
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {str(e)}")
        plt.text(0.5, 0.5, 'Feature importance visualization not available for this model', 
                 ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_shap_summary_plot(model, X_train_processed, X_test_processed, feature_names, task_type):
    """Generate a SHAP summary plot."""
    try:
        plt.figure(figsize=(12, 8))
        
        X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
        X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
        
        max_samples = min(100, X_test_dense.shape[0])
        X_sample = X_test_dense[:max_samples]
        
        if task_type == 'classification':
            if hasattr(model, 'estimators_'):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_sample)
                    
                    if hasattr(shap_values, 'values'):
                        shap.summary_plot(shap_values.values, X_sample, 
                                          feature_names=feature_names, show=False)
                    else:
                        if isinstance(shap_values, list):
                            shap.summary_plot(shap_values[1], X_sample, 
                                              feature_names=feature_names, show=False)
                        else:
                            shap.summary_plot(shap_values, X_sample, 
                                              feature_names=feature_names, show=False)
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {str(e)}, trying Kernel explainer")
                    background = shap.kmeans(X_train_dense, 10)
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_sample)
                    
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1], X_sample, 
                                          feature_names=feature_names, show=False)
                    else:
                        shap.summary_plot(shap_values, X_sample, 
                                          feature_names=feature_names, show=False)
            else:
                background = shap.kmeans(X_train_dense, 10)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], X_sample, 
                                      feature_names=feature_names, show=False)
                else:
                    shap.summary_plot(shap_values, X_sample, 
                                      feature_names=feature_names, show=False)
        else:
            if hasattr(model, 'estimators_'):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_sample)
                    
                    if hasattr(shap_values, 'values'):
                        shap.summary_plot(shap_values.values, X_sample, 
                                          feature_names=feature_names, show=False)
                    else:
                        shap.summary_plot(shap_values, X_sample, 
                                          feature_names=feature_names, show=False)
                except:
                    background = shap.kmeans(X_train_dense, 10)
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X_sample)
                    shap.summary_plot(shap_values, X_sample, 
                                      feature_names=feature_names, show=False)
            else:
                background = shap.kmeans(X_train_dense, 10)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample)
                shap.summary_plot(shap_values, X_sample, 
                                  feature_names=feature_names, show=False)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating SHAP summary plot: {str(e)}")
        plt.close()
        return None

def generate_shap_force_plot(model, X_train_processed, X_test_processed, feature_names, index, task_type):
    """Generate a SHAP force plot for individual instance explanation."""
    try:
        plt.figure(figsize=(14, 4))
        
        X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
        X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
        
        X_instance = X_test_dense[index:index+1]
        
        if task_type == 'classification':
            if hasattr(model, 'estimators_'):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_instance)
                    
                    if hasattr(shap_values, 'values'):
                        plt.close() 
                        fig = plt.figure(figsize=(14, 4))
                        ax = fig.add_subplot(111)
                        shap.plots.waterfall(shap_values[0], max_display=20, show=False)
                    else:
                        if isinstance(shap_values, list):
                            shap.force_plot(explainer.expected_value[1], shap_values[1][0], 
                                            features=X_instance[0], feature_names=feature_names, 
                                            matplotlib=True, show=False)
                        else:
                            shap.force_plot(explainer.expected_value, shap_values[0], 
                                            features=X_instance[0], feature_names=feature_names, 
                                            matplotlib=True, show=False)
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {str(e)}, trying KernelExplainer")
                    background = shap.kmeans(X_train_dense, 10)
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_instance)
                    
                    if isinstance(shap_values, list):
                        shap.force_plot(explainer.expected_value[1], shap_values[1][0], 
                                        features=X_instance[0], feature_names=feature_names, 
                                        matplotlib=True, show=False)
                    else:
                        shap.force_plot(explainer.expected_value, shap_values[0], 
                                        features=X_instance[0], feature_names=feature_names, 
                                        matplotlib=True, show=False)
            else:
                background = shap.kmeans(X_train_dense, 10)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_instance)
                
                if isinstance(shap_values, list):
                    shap.force_plot(explainer.expected_value[1], shap_values[1][0], 
                                    features=X_instance[0], feature_names=feature_names, 
                                    matplotlib=True, show=False)
                else:
                    shap.force_plot(explainer.expected_value, shap_values[0], 
                                    features=X_instance[0], feature_names=feature_names, 
                                    matplotlib=True, show=False)
        else:
            if hasattr(model, 'estimators_'):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X_instance)
                    
                    if hasattr(shap_values, 'values'):
                        plt.close()
                        fig = plt.figure(figsize=(14, 4))
                        ax = fig.add_subplot(111)
                        shap.plots.waterfall(shap_values[0], max_display=20, show=False)
                    else:
                        shap.force_plot(explainer.expected_value, shap_values[0], 
                                        features=X_instance[0], feature_names=feature_names, 
                                        matplotlib=True, show=False)
                except:
                    background = shap.kmeans(X_train_dense, 10)
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X_instance)
                    shap.force_plot(explainer.expected_value, shap_values[0], 
                                    features=X_instance[0], feature_names=feature_names, 
                                    matplotlib=True, show=False)
            else:
                background = shap.kmeans(X_train_dense, 10)
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_instance)
                shap.force_plot(explainer.expected_value, shap_values[0], 
                                features=X_instance[0], feature_names=feature_names, 
                                matplotlib=True, show=False)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.tight_layout()
        plt.close()
        buf.seek(0)
        
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating SHAP force plot: {str(e)}")
        plt.close()
        return None

def run_automl_pipeline(df, target_col, n_trials=30):
    """Main AutoML pipeline function."""
    try:
        data = preprocess_data(df, target_col)
        X_train = data['X_train_processed']
        y_train = data['y_train']
        X_test = data['X_test_processed']
        y_test = data['y_test']
        feature_names = data['feature_names']
        label_encoder = data['label_encoder']
        
        task_type = determine_task_type(y_train)
        logger.info(f"Task type determined: {task_type}")
        
        def wrapped_objective(trial):
            return objective(trial, X_train, y_train, X_test, y_test, task_type)

        study = optuna.create_study(direction='maximize')
        study.optimize(wrapped_objective, n_trials=n_trials)

        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        
        if task_type == 'classification':
            if best_params['model'] == 'logistic':
                best_model = LogisticRegression(
                    C=best_params['logreg_C'], 
                    solver=best_params['solver'],
                    penalty=best_params['penalty'],
                    max_iter=1000, 
                    random_state=42
                )
            else:
                best_model = RandomForestClassifier(
                    n_estimators=best_params['rf_n_estimators'],
                    max_depth=best_params['rf_max_depth'],
                    min_samples_split=best_params['min_samples_split'],
                    min_samples_leaf=best_params['min_samples_leaf'],
                    random_state=42
                )
        else:
            if best_params['model'] == 'linear':
                best_model = LinearRegression()
            else:
                best_model = RandomForestRegressor(
                    n_estimators=best_params['rf_n_estimators'],
                    max_depth=best_params['rf_max_depth'],
                    min_samples_split=best_params['min_samples_split'],
                    min_samples_leaf=best_params['min_samples_leaf'],
                    random_state=42
                )
        
        best_model.fit(X_train, y_train)
        
        if task_type == 'classification':
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_ if label_encoder else None)
            conf_matrix = generate_confusion_matrix_plot(
                y_test, y_pred, 
                class_names=label_encoder.classes_ if label_encoder else None
            )
            
            roc_curve_plot = None
            if y_prob is not None:
                roc_curve_plot = generate_roc_curve_plot(y_test, y_prob)
            
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'roc_curve': roc_curve_plot
            }
        else:
            y_pred = best_model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            residual_plot = generate_residual_plot(y_test, y_pred)
            actual_vs_predicted = generate_actual_vs_predicted_plot(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'residual_plot': residual_plot,
                'actual_vs_predicted': actual_vs_predicted
            }
        
        feature_importance = generate_feature_importance_plot(best_model, feature_names)
        shap_summary = generate_shap_summary_plot(
            best_model, X_train, X_test, feature_names, task_type
        )
        
        return {
            'best_model': best_model,
            'preprocessor': data['preprocessor'],
            'metrics': metrics,
            'feature_importance': feature_importance,
            'shap_summary': shap_summary,
            'feature_names': feature_names,
            'task_type': task_type,
            'label_encoder': label_encoder,
            'X_test_processed': X_test,
            'y_test': y_test
        }
    except Exception as e:
        logger.error(f"Error in AutoML pipeline: {str(e)}")
        raise