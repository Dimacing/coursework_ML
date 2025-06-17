import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

from src import config
from src.data_loader import load_and_preprocess_data
from src.utils import evaluate_classification, perform_grid_search, plot_feature_importance, plot_roc_curve, plot_precision_recall_curve


def train_si_gt_8_classification():
    RAW_TARGET_FOR_THRESHOLD = config.TARGET_SI
    THRESHOLD_VALUE = config.SI_THRESHOLD_CLASSIFICATION
    TASK_NAME = f"SI_gt_{str(THRESHOLD_VALUE).replace('.', 'p')}_Classification"
    print(f"--- Starting {TASK_NAME} ---")

    X_train, Y_train, X_test, Y_test, feature_cols, scaler, imputer, _ = load_and_preprocess_data(
        apply_log_transform_targets=True,
        drop_highly_correlated=True,
        drop_low_variance=True
    )

    if X_train is None or RAW_TARGET_FOR_THRESHOLD not in Y_train.columns:
        print(f"Could not load data or raw target '{RAW_TARGET_FOR_THRESHOLD}' missing. Aborting.")
        return None, {}

    y_train_raw = Y_train[RAW_TARGET_FOR_THRESHOLD]
    y_test_raw = Y_test[RAW_TARGET_FOR_THRESHOLD]

    y_train = (y_train_raw > THRESHOLD_VALUE).astype(int)
    y_test = (y_test_raw > THRESHOLD_VALUE).astype(int)

    print(f"Train class distribution ({RAW_TARGET_FOR_THRESHOLD} > {THRESHOLD_VALUE}):\n{y_train.value_counts(normalize=True)}")
    if y_train.nunique() < 2:
        print(f"Warning: Training target for {TASK_NAME} has <2 unique classes.")

    minority_count = (y_train == 1).sum()
    majority_count = (y_train == 0).sum()
    scale_pos_weight_val = majority_count / minority_count if minority_count > 0 else 1
    print(f"Calculated scale_pos_weight: {scale_pos_weight_val:.2f}")

    models_params = {
        "LogisticRegression": (
            LogisticRegression(random_state=config.RANDOM_STATE, solver='liblinear', class_weight='balanced', max_iter=1000),
            {'C': [0.1, 1, 10]}),
        "KNeighborsClassifier": (KNeighborsClassifier(), {'n_neighbors': [5, 7, 9]}),
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced_subsample', n_jobs=-1),
            {'n_estimators': [100, 200], 'max_depth': [10, None]}),
        "GradientBoostingClassifier": (
            GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'),
            {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31, 50]}),
        "AdaBoostClassifier": (
            AdaBoostClassifier(random_state=config.RANDOM_STATE),
            {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
        "XGBClassifier": (
            XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'scale_pos_weight': [scale_pos_weight_val]}),
        "LGBMClassifier": (
            LGBMClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=-1),
            {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}),
    }

    best_estimators = {}
    cv_results_summary = {}
    test_results_summary = {}
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    for name, (model, params) in models_params.items():
        print(f"\n--- Training and Tuning {name} for {TASK_NAME} ---")
        try:
            best_estimator, best_cv_score = perform_grid_search(
                X_train, y_train, model, params, cv_strategy=skf, scoring='roc_auc'
            )
            best_estimators[name] = best_estimator
            cv_results_summary[name] = {"Best CV ROC_AUC": best_cv_score}

            y_pred_test = best_estimator.predict(X_test)
            y_proba_test = best_estimator.predict_proba(X_test)[:, 1] if hasattr(best_estimator, "predict_proba") else None

            current_test_results = evaluate_classification(y_test, y_pred_test, y_proba_test, model_name=name, task_name=TASK_NAME)
            test_results_summary[name] = current_test_results

            plot_feature_importance(best_estimator, feature_cols, model_name=name, task_name=TASK_NAME)
            if y_proba_test is not None:
                plot_roc_curve(y_test, y_proba_test, model_name=name, task_name=TASK_NAME)
                plot_precision_recall_curve(y_test, y_proba_test, model_name=name, task_name=TASK_NAME)

            joblib.dump(best_estimator, config.MODEL_DIR / f"{TASK_NAME.lower()}_{name.lower()}_best_model.joblib")
        except Exception as e:
            print(f"ERROR training {name} for {TASK_NAME}: {e}")
            cv_results_summary[name] = {"Best CV ROC_AUC": np.nan}
            test_results_summary[name] = {m: np.nan for m in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]}

    results_df = pd.DataFrame(test_results_summary).T.drop(columns=['CM'], errors='ignore')
    results_df = results_df.join(pd.DataFrame(cv_results_summary).T)
    results_df = results_df.sort_values(by='PR_AUC', ascending=False)

    print(f"\n\n--- Model Comparison for {TASK_NAME} (Sorted by Test PR_AUC) ---")
    print(results_df)
    results_df.to_excel(config.REPORT_DIR_ML / f"{TASK_NAME.lower()}_model_comparison.xlsx")

    if not results_df.empty:
        best_model_name_on_test = results_df['PR_AUC'].idxmax(skipna=True)
        if pd.notna(best_model_name_on_test):
            print(f"\nBest model for {TASK_NAME} based on Test PR_AUC: {best_model_name_on_test} (PR_AUC: {results_df.loc[best_model_name_on_test, 'PR_AUC']:.4f})")

    print(f"--- {TASK_NAME} Complete ---")
    return results_df, best_estimators


if __name__ == '__main__':
    train_si_gt_8_classification()