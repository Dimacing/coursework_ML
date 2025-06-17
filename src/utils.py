import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import re
from src import config

def clean_col_names(df_input):
    df = df_input.copy()
    rename_map = {
        'Unnamed: 0': 'ID_original_index',
        'IC50, mM': config.TARGET_IC50,
        'CC50, mM': config.TARGET_CC50,
        'SI': config.TARGET_SI
    }
    df.rename(columns=rename_map, inplace=True)
    new_cols_dict = {}
    for col in df.columns:
        if col in rename_map.values():
            new_cols_dict[col] = col
            continue
        new_name = str(col).strip()
        new_name = re.sub(r'[,\s]+', '_', new_name)
        new_name = re.sub(r'[µmM%/\-\(\)]+', '', new_name)
        new_name = re.sub(r'[^A-Za-z0-9_]+', '', new_name)
        new_name = new_name.replace('__', '_')
        new_name = new_name.strip('_')
        original_new_name = new_name
        counter = 1
        while new_name in new_cols_dict.values():
            new_name = f"{original_new_name}_{counter}"
            counter += 1
        new_cols_dict[col] = new_name
    df.columns = list(new_cols_dict.values())
    return df

def calculate_si(df, ic50_col, cc50_col):
    ic50_safe = df[ic50_col].copy()
    cc50_safe = df[cc50_col].copy()
    ic50_safe[ic50_safe <= 0] = np.nan
    cc50_safe[cc50_safe <= 0] = np.nan
    si = cc50_safe / ic50_safe
    return si

def transform_to_p_value(series_mM, epsilon=config.EPSILON):
    series_M = series_mM * 1e-3
    series_M_safe = series_M.copy()
    series_M_safe[series_M_safe <= 0] = epsilon
    return -np.log10(series_M_safe)

def transform_to_log(series, use_log1p=True, epsilon=config.EPSILON):
    series_safe = series.copy()
    if use_log1p:
        series_safe[series_safe < 0] = 0
        return np.log1p(series_safe)
    else:
        series_safe[series_safe <= 0] = epsilon
        return np.log10(series_safe)

def evaluate_regression(y_true, y_pred, model_name="Model"):
    y_true_finite = y_true[np.isfinite(y_pred) & np.isfinite(y_true)]
    y_pred_finite = y_pred[np.isfinite(y_pred) & np.isfinite(y_true)]
    if len(y_true_finite) == 0:
        print(f"--- {model_name} Performance ---")
        print("No finite values to evaluate regression.")
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MedAE": np.nan}
    r2 = r2_score(y_true_finite, y_pred_finite)
    mae = mean_absolute_error(y_true_finite, y_pred_finite)
    rmse = mean_squared_error(y_true_finite, y_pred_finite, squared=False)
    medae = np.median(np.abs(y_true_finite - y_pred_finite))
    print(f"--- {model_name} Performance ---")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MedAE: {medae:.4f}")
    print("-----------------------------------")
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MedAE": medae}

def evaluate_classification(y_true, y_pred, y_proba, model_name="Model", average='binary', task_name="classification_task"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0, average=average if y_true.nunique() > 2 else 'binary')
    recall = recall_score(y_true, y_pred, zero_division=0, average=average if y_true.nunique() > 2 else 'binary')
    f1 = f1_score(y_true, y_pred, zero_division=0, average=average if y_true.nunique() > 2 else 'binary')
    roc_auc_val = "N/A"
    pr_auc_val = "N/A"
    if y_proba is not None:
        if y_true.nunique() == 2:
            roc_auc_val = roc_auc_score(y_true, y_proba)
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            pr_auc_val = auc(rec, prec)
    print(f"--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if y_proba is not None:
         print(f"ROC AUC Score: {roc_auc_val if isinstance(roc_auc_val, str) else roc_auc_val:.4f}")
         print(f"PR AUC Score: {pr_auc_val if isinstance(pr_auc_val, str) else pr_auc_val:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix - {model_name}\n{task_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    cm_filename = f"cm_{task_name.lower()}_{model_name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
    fig.savefig(config.REPORT_DIR_ML / cm_filename)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {config.REPORT_DIR_ML / cm_filename}")
    print("-----------------------------------")
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "ROC_AUC": roc_auc_val, "PR_AUC": pr_auc_val, "CM": cm}

def perform_grid_search(X_train, y_train, model, param_grid, cv_strategy, scoring='neg_mean_squared_error'):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_strategy,
                               scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_score_

def plot_feature_importance(model, feature_names, model_name="Model", task_name="task", top_n=20):
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0] if isinstance(model.coef_, list) or model.coef_.ndim > 1 else model.coef_)
    if importances is not None:
        indices = np.argsort(importances)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.3)))
        ax.set_title(f"Top {top_n} Feature Importances - {model_name}\n{task_name}")
        ax.barh(range(len(indices)), importances[indices][::-1], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices][::-1])
        ax.invert_yaxis()
        ax.set_xlabel('Importance / Absolute Coefficient Value')
        plt.tight_layout()
        plot_filename = f"fi_{task_name.lower()}_{model_name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
        fig.savefig(config.REPORT_DIR_ML / plot_filename)
        plt.close(fig)
        print(f"Feature importance plot saved to {config.REPORT_DIR_ML / plot_filename}")
    else:
        print(f"Model {model_name} does not have 'feature_importances_' or 'coef_' attribute.")

def plot_regression_diagnostics(y_true, y_pred, model_name="Model", task_name="task"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].scatter(y_pred, y_true, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Actual Values')
    axes[0].set_title(f'Actual vs. Predicted - {model_name}\n{task_name}')
    axes[0].grid(True)
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].hlines(0, y_pred.min(), y_pred.max(), colors='red', linestyles='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title(f'Residuals Plot - {model_name}\n{task_name}')
    axes[1].grid(True)
    plt.tight_layout()
    diag_filename = f"diag_{task_name.lower()}_{model_name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
    fig.savefig(config.REPORT_DIR_ML / diag_filename)
    plt.close(fig)
    print(f"Regression diagnostics plot saved to {config.REPORT_DIR_ML / diag_filename}")

def plot_roc_curve(y_true, y_proba, model_name="Model", task_name="task"):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_val:0.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic - {model_name}\n{task_name}')
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    roc_filename = f"roc_{task_name.lower()}_{model_name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
    fig.savefig(config.REPORT_DIR_ML / roc_filename)
    plt.close(fig)
    print(f"ROC curve plot saved to {config.REPORT_DIR_ML / roc_filename}")

def plot_precision_recall_curve(y_true, y_proba, model_name="Model", task_name="task"):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_val = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(rec, prec, color='blue', lw=2, label=f'PR curve (area = {pr_auc_val:0.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f'Precision-Recall Curve - {model_name}\n{task_name}')
    ax.legend(loc="lower left")
    ax.grid(True)
    plt.tight_layout()
    pr_filename = f"pr_curve_{task_name.lower()}_{model_name.lower().replace(' ', '_').replace('(','').replace(')','')}.png"
    fig.savefig(config.REPORT_DIR_ML / pr_filename)
    plt.close(fig)
    print(f"Precision-Recall curve plot saved to {config.REPORT_DIR_ML / pr_filename}")
