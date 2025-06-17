import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src import config
from src.utils import clean_col_names, calculate_si, transform_to_p_value, transform_to_log

def load_and_preprocess_data(apply_log_transform_targets=True,
                             drop_highly_correlated=True, corr_threshold=0.90,
                             drop_low_variance=True, var_threshold=1e-4):
    print("--- Starting Data Loading and Preprocessing ---")
    try:
        df = pd.read_excel(config.RAW_DATA_FILE, engine='openpyxl')
        print(f"Successfully loaded raw data. Initial shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {config.RAW_DATA_FILE}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'ID_original_index'}, inplace=True)
        original_ids = df['ID_original_index'].copy()
    else:
        original_ids = pd.Series(range(len(df)), name='temp_id')

    df = clean_col_names(df)

    print(f"Column names cleaned. Shape: {df.shape}")

    cols_to_drop_current = []
    if 'ID_original_index' in df.columns:
        cols_to_drop_current.append('ID_original_index')
    if 'SPS' in df.columns:
        cols_to_drop_current.append('SPS')

    for col_orig in config.FEATURES_TO_DROP_INITIAL:
        if col_orig not in ['Unnamed: 0', 'SPS']:
            temp_df_for_name_find = pd.DataFrame(columns=[col_orig])
            cleaned_name = clean_col_names(temp_df_for_name_find).columns[0]
            if cleaned_name in df.columns and cleaned_name not in cols_to_drop_current:
                cols_to_drop_current.append(cleaned_name)

    if cols_to_drop_current:
        df.drop(columns=cols_to_drop_current, inplace=True, errors='ignore')
        print(f"Dropped initial columns: {cols_to_drop_current}. Shape: {df.shape}")

    for target_col in [config.TARGET_IC50, config.TARGET_CC50]:
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        else:
            print(f"Warning: Target column '{target_col}' not found.")

    if config.TARGET_IC50 in df.columns and config.TARGET_CC50 in df.columns:
        df[config.TARGET_SI] = calculate_si(df, config.TARGET_IC50, config.TARGET_CC50)
        print(f"SI calculated. NaN count in SI: {df[config.TARGET_SI].isnull().sum()}")
    else:
        df[config.TARGET_SI] = np.nan

    df_current_index = df.index.copy()
    essential_targets = [config.TARGET_IC50, config.TARGET_CC50, config.TARGET_SI]
    present_essential_targets = [t for t in essential_targets if t in df.columns]

    if present_essential_targets:
        nan_mask_essential = df[present_essential_targets].isnull().any(axis=1)
        df = df[~nan_mask_essential]
        original_ids = original_ids.loc[df.index]
        print(f"Dropped {nan_mask_essential.sum()} rows due to NaN in any of {present_essential_targets}. Shape: {df.shape}")

    if df.empty:
        print("DataFrame is empty after dropping NaNs in essential targets. Aborting.")
        return None, None, None, None, None

    if apply_log_transform_targets:
        if config.TARGET_IC50 in df.columns:
            df[config.LOG_TARGET_IC50] = transform_to_p_value(df[config.TARGET_IC50])
        if config.TARGET_CC50 in df.columns:
            df[config.LOG_TARGET_CC50] = transform_to_p_value(df[config.TARGET_CC50])
        if config.TARGET_SI in df.columns:
            df[config.LOG_TARGET_SI] = transform_to_log(df[config.TARGET_SI], use_log1p=True)

        log_targets_list = [config.LOG_TARGET_IC50, config.LOG_TARGET_CC50, config.LOG_TARGET_SI]
        present_log_targets = [t for t in log_targets_list if t in df.columns]

        if present_log_targets:
            nan_mask_log = df[present_log_targets].isnull().any(axis=1)
            df = df[~nan_mask_log]
            original_ids = original_ids.loc[df.index]
            print(f"Dropped {nan_mask_log.sum()} rows due to NaN in any log-transformed targets. Shape: {df.shape}")

    if df.empty:
        print("DataFrame is empty after dropping NaNs in log-transformed targets. Aborting.")
        return None, None, None, None, None

    all_target_cols_final = [
        config.TARGET_IC50, config.TARGET_CC50, config.TARGET_SI,
        config.LOG_TARGET_IC50, config.LOG_TARGET_CC50, config.LOG_TARGET_SI
    ]
    present_all_target_cols = [col for col in all_target_cols_final if col in df.columns]
    feature_columns = [col for col in df.columns if col not in present_all_target_cols]

    X_data = df[feature_columns].copy()
    Y_data = df[present_all_target_cols].copy()

    X_train, X_test, Y_train, Y_test, ids_train, ids_test = train_test_split(
        X_data, Y_data, original_ids,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    print(f"Data split into train/test. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    imputer = SimpleImputer(strategy='median')
    X_train_imputed_array = imputer.fit_transform(X_train)
    X_train_imputed = pd.DataFrame(X_train_imputed_array, columns=X_train.columns, index=X_train.index)

    if drop_low_variance:
        variances = X_train_imputed.var()
        low_variance_cols = variances[variances < var_threshold].index.tolist()
        if low_variance_cols:
            X_train_imputed.drop(columns=low_variance_cols, inplace=True)
            print(f"Dropped {len(low_variance_cols)} low variance columns: {low_variance_cols}")
            feature_columns = X_train_imputed.columns.tolist()
        else:
            print("No low variance columns found to drop.")

    if drop_highly_correlated:
        corr_matrix = X_train_imputed.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = set()
        for column in upper.columns:
            if column not in to_drop_corr:
                highly_correlated_with_column = upper[upper[column] > corr_threshold].index
                for correlated_col in highly_correlated_with_column:
                    if correlated_col not in to_drop_corr:
                        to_drop_corr.add(correlated_col)
        if to_drop_corr:
            X_train_imputed.drop(columns=list(to_drop_corr), inplace=True)
            print(f"Dropped {len(to_drop_corr)} highly correlated columns: {list(to_drop_corr)[:5]}...")
            feature_columns = X_train_imputed.columns.tolist()
        else:
            print("No highly correlated columns found to drop.")

    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train_imputed)
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=X_train_imputed.columns, index=X_train_imputed.index)

    X_test_imputed_array = imputer.transform(X_test[X_train.columns])
    X_test_imputed = pd.DataFrame(X_test_imputed_array, columns=X_train.columns, index=X_test.index)

    if drop_low_variance and 'low_variance_cols' in locals() and low_variance_cols:
        X_test_imputed.drop(columns=low_variance_cols, inplace=True, errors='ignore')
    if drop_highly_correlated and 'to_drop_corr' in locals() and to_drop_corr:
        X_test_imputed.drop(columns=list(to_drop_corr), inplace=True, errors='ignore')

    X_test_scaled_array = scaler.transform(X_test_imputed)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=X_train_scaled.columns, index=X_test_imputed.index)

    final_feature_columns = X_train_scaled.columns.tolist()
    print(f"Final number of features after preprocessing: {len(final_feature_columns)}")

    print("--- Data Loading and Preprocessing Complete ---")
    return X_train_scaled, Y_train, X_test_scaled, Y_test, final_feature_columns, scaler, imputer, ids_test

if __name__ == '__main__':
    X_train_final, Y_train_final, X_test_final, Y_test_final, features_final, _, _, test_ids = load_and_preprocess_data()
    if X_train_final is not None:
        print("\n--- Processed Training Data Sample (Head) ---")
        print(X_train_final.head())
        print(Y_train_final.head())
        print("\n--- Processed Test Data Sample (Head) ---")
        print(X_test_final.head())
        print(Y_test_final.head())
        print("\nTest IDs sample: {test_ids.head().tolist()}")
        print("\nNumber of final features: {len(features_final)}")
        print("\nFinal features list (first 10):", features_final[:10])