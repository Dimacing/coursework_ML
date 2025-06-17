import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import joblib
from src import config
from src.data_loader import load_and_preprocess_data
from src.utils import evaluate_regression, perform_grid_search, plot_feature_importance, plot_regression_diagnostics

def train_logsi_regression():
    TARGET_NAME_FOR_MODEL = config.LOG_TARGET_SI
    TASK_NAME = "logSI_Regression"
    print(f"--- Starting {TASK_NAME} ---")

    X_train, Y_train, X_test, Y_test, feature_cols, scaler, imputer, _ = load_and_preprocess_data(
        apply_log_transform_targets=True,
        drop_highly_correlated=True,
        drop_low_variance=True
    )

    if X_train is None or TARGET_NAME_FOR_MODEL not in Y_train.columns:
        print(f"Could not load data or target '{TARGET_NAME_FOR_MODEL}' is missing. Aborting.")
        return None, {}

    y_train = Y_train[TARGET_NAME_FOR_MODEL]
    y_test = Y_test[TARGET_NAME_FOR_MODEL]

    print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    models_params = {
        "Ridge": (Ridge(random_state=config.RANDOM_STATE), {'alpha': [0.1, 1.0, 10.0, 100.0]}),
        "KNeighborsRegressor": (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 9]}),
        "RandomForestRegressor": (RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1), {
            'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_leaf': [2, 5]
        }),
        "GradientBoostingRegressor": (GradientBoostingRegressor(random_state=config.RANDOM_STATE), {
            'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]
        }),
        "HistGradientBoostingRegressor": (HistGradientBoostingRegressor(random_state=config.RANDOM_STATE), {
            'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31, 50]
        }),
        "AdaBoostRegressor": (AdaBoostRegressor(random_state=config.RANDOM_STATE), {
            'n_estimators': [50, 100], 'learning_rate': [0.1, 0.5, 1.0]
        }),
        "XGBRegressor": (
            XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, objective='reg:squarederror', eval_metric='rmse'), {
                'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]
            }
        ),
        "LGBMRegressor": (LGBMRegressor(random_state=config.RANDOM_STATE, n_jobs=-1), {
            'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 50]
        }),
    }

    estimators_for_stacking = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE, max_depth=10, min_samples_leaf=5)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=config.RANDOM_STATE, objective='reg:squarederror')),
    ]
    stacking_regressor = StackingRegressor(
        estimators=estimators_for_stacking,
        final_estimator=LinearRegression(),
        cv=3
    )
    models_params["StackingRegressor"] = (stacking_regressor, {})

    best_estimators = {}
    cv_results_summary = {}
    test_results_summary = {}
    kf = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    for name, (model, params) in models_params.items():
        print(f"\n--- Training and Tuning {name} for {TASK_NAME} ---")
        try:
            if name == "StackingRegressor" and not params:
                model.fit(X_train, y_train)
                best_estimator = model
                cv_results_summary[name] = {"Best CV R2": "N/A (Stacking)"}
            else:
                best_estimator, best_cv_score = perform_grid_search(
                    X_train, y_train, model, params, cv_strategy=kf, scoring='r2'
                )
                cv_results_summary[name] = {"Best CV R2": best_cv_score}

            best_estimators[name] = best_estimator
            y_pred_test = best_estimator.predict(X_test)
            current_test_results = evaluate_regression(y_test, y_pred_test, model_name=name)
            test_results_summary[name] = current_test_results

            plot_feature_importance(best_estimator, feature_cols, model_name=name, task_name=TASK_NAME)
            plot_regression_diagnostics(y_test, y_pred_test, model_name=name, task_name=TASK_NAME)

            joblib.dump(best_estimator, config.MODEL_DIR / f"{TASK_NAME.lower()}_{name.lower()}_best_model.joblib")

        except Exception as e:
            print(f"ERROR training {name} for {TASK_NAME}: {e}")
            cv_results_summary[name] = {"Best CV R2": np.nan}
            test_results_summary[name] = {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MedAE": np.nan}

    results_df = pd.DataFrame(test_results_summary).T.join(pd.DataFrame(cv_results_summary).T)
    results_df = results_df.sort_values(by='R2', ascending=False)

    print(f"\n\n--- Model Comparison for {TASK_NAME} (Sorted by Test R2) ---")
    print(results_df)
    results_df.to_excel(config.REPORT_DIR_ML / f"{TASK_NAME.lower()}_model_comparison.xlsx")

    if not results_df.empty:
        best_model_name_on_test = results_df['R2'].idxmax()
        print(
            f"\nBest model for {TASK_NAME} based on Test R2: {best_model_name_on_test} (R2: {results_df.loc[best_model_name_on_test, 'R2']:.4f})")

    print(f"--- {TASK_NAME} Complete ---")
    return results_df, best_estimators

if __name__ == '__main__':
    train_logsi_regression()
