import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.impute import SimpleImputer
import re
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ""
RAW_DATA_FILE = DATA_DIR / "mo.xlsx"
REPORT_DIR_EDA = BASE_DIR / "report_eda"
REPORT_DIR_EDA.mkdir(parents=True, exist_ok=True)

FINAL_TARGET_IC50 = "IC50_mM"
FINAL_TARGET_CC50 = "CC50_mM"
FINAL_TARGET_SI = "SI"


def simplified_clean_col_names(df_input):
    df = df_input.copy()
    rename_map = {
        'Unnamed: 0': 'ID_original_index',
        'IC50, mM': FINAL_TARGET_IC50,
        'CC50, mM': FINAL_TARGET_CC50,
        'SI': FINAL_TARGET_SI
    }
    df.rename(columns=rename_map, inplace=True)

    new_cols = {}
    for col in df.columns:
        if col in rename_map.values():
            new_cols[col] = col
            continue
        new_name = str(col).strip()
        new_name = re.sub(r'[,\s]+', '_', new_name)
        new_name = re.sub(r'[^A-Za-z0-9_]+', '', new_name)
        new_name = new_name.replace('__', '_').strip('_')
        original_new_name = new_name
        counter = 1
        while new_name in new_cols.values():
            new_name = f"{original_new_name}_{counter}"
            counter += 1
        new_cols[col] = new_name
    df.rename(columns=new_cols, inplace=True)
    return df


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")


def run_direct_eda():
    print("--- Starting Direct EDA ---")
    try:
        df = pd.read_excel(RAW_DATA_FILE, engine='openpyxl')
        print(f"Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}");
        return

    df = simplified_clean_col_names(df.copy())

    if 'ID_original_index' in df.columns:
        df = df.drop(columns=['ID_original_index'])
        print("Dropped 'ID_original_index'.")

    targets_to_convert = [FINAL_TARGET_IC50, FINAL_TARGET_CC50, FINAL_TARGET_SI]
    for target_col in targets_to_convert:
        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        else:
            print(f"Warning: Expected target column '{target_col}' not found after cleaning.")

    if FINAL_TARGET_IC50 in df.columns and FINAL_TARGET_CC50 in df.columns:
        ic50_safe = df[FINAL_TARGET_IC50].copy()
        cc50_safe = df[FINAL_TARGET_CC50].copy()
        ic50_safe[ic50_safe <= 0] = np.nan
        cc50_safe[cc50_safe <= 0] = np.nan
        df[FINAL_TARGET_SI] = cc50_safe / ic50_safe
        print(f"Recalculated '{FINAL_TARGET_SI}'.")

    eda_target_cols = [t for t in [FINAL_TARGET_IC50, FINAL_TARGET_CC50, FINAL_TARGET_SI] if
                       t in df.columns and not df[t].isnull().all()]

    if eda_target_cols:
        print("\n--- Target Variable Analysis ---")
        for target in eda_target_cols:
            print(f"\n{target} - Statistics:\n{df[target].describe()}")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(df[target].dropna(), kde=True, ax=axes[0], bins=30).set_title(f'Distribution of {target}')
            sns.boxplot(y=df[target].dropna(), ax=axes[1]).set_title(f'Boxplot of {target}')
            plt.tight_layout();
            fig.savefig(REPORT_DIR_EDA / f"dist_{target}.png");
            plt.close(fig)  # Закрываем фигуру после сохранения

            positive_data = df[target].dropna()
            positive_data = positive_data[positive_data > 0]
            if not positive_data.empty:
                fig_log, axes_log = plt.subplots(1, 2, figsize=(12, 4))
                sns.histplot(np.log1p(positive_data), kde=True, ax=axes_log[0], bins=30).set_title(
                    f'Distribution of log1p({target})')
                sns.boxplot(y=np.log1p(positive_data), ax=axes_log[1]).set_title(f'Boxplot of log1p({target})')
                plt.tight_layout();
                fig_log.savefig(REPORT_DIR_EDA / f"dist_log1p_{target}.png");
                plt.close(fig_log)  # Закрываем фигуру

    if not df.empty:
        print("\n--- Missing Values ---")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        missing_percent = (missing / len(df)) * 100
        missing_data = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_percent})
        print(missing_data.head(15))
        if not missing_data.empty:
            fig_missing = plt.figure(figsize=(12, 6))  # Сохраняем ссылку на фигуру
            sns.barplot(x=missing_data.index[:20], y='Percentage', data=missing_data.head(20))
            plt.xticks(rotation=90);
            plt.title('Top 20 Features with Missing Values (%)');
            plt.tight_layout()
            fig_missing.savefig(REPORT_DIR_EDA / "missing_values.png");
            plt.close(fig_missing)  # Закрываем фигуру

    print("\n--- Feature Correlation Analysis ---")
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    features_for_corr = [f for f in numerical_features if f not in eda_target_cols]

    if len(features_for_corr) > 1:
        df_imputed_for_corr = df[features_for_corr].copy()
        for col in df_imputed_for_corr.columns:
            if df_imputed_for_corr[col].dtype == 'object':
                df_imputed_for_corr[col] = pd.to_numeric(df_imputed_for_corr[col], errors='coerce')
        df_imputed_for_corr.dropna(axis=1, how='all', inplace=True)

        if df_imputed_for_corr.isnull().any().any() and not df_imputed_for_corr.empty:
            imputer_corr = SimpleImputer(strategy='median')
            df_imputed_for_corr = pd.DataFrame(imputer_corr.fit_transform(df_imputed_for_corr),
                                               columns=df_imputed_for_corr.columns)

        if not df_imputed_for_corr.empty and len(df_imputed_for_corr.columns) > 1:
            correlation_matrix = df_imputed_for_corr.corr()
            num_features_in_corr = len(df_imputed_for_corr.columns)
            if num_features_in_corr <= 50:
                fig_corr_matrix = plt.figure(
                    figsize=(max(12, num_features_in_corr * 0.3), max(10, num_features_in_corr * 0.3)))
                sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Feature Correlation Matrix');
                plt.tight_layout();
                fig_corr_matrix.savefig(REPORT_DIR_EDA / "feature_corr_matrix.png");
                plt.close(fig_corr_matrix)

            upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [(r, c, upper.loc[r, c]) for r in upper.index for c in upper.columns if
                               abs(upper.loc[r, c]) > 0.90]
            if high_corr_pairs:
                print(f"\nHighly correlated feature pairs (abs_corr > 0.90), sorted (top 10):")
                sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
                for pair in sorted_pairs[:10]:
                    print(f"- {pair[0]} & {pair[1]}: {pair[2]:.3f}")
            else:
                print("No feature pairs found with absolute correlation > 0.90.")

            if eda_target_cols:
                df_targets_imputed = df[eda_target_cols + features_for_corr].copy()
                for col in df_targets_imputed.columns:
                    if df_targets_imputed[col].dtype == 'object':
                        df_targets_imputed[col] = pd.to_numeric(df_targets_imputed[col], errors='coerce')
                df_targets_imputed.dropna(axis=1, how='all', inplace=True)

                if df_targets_imputed.isnull().any().any() and not df_targets_imputed.empty:
                    imputer_targets_corr = SimpleImputer(strategy='median')
                    df_targets_imputed = pd.DataFrame(imputer_targets_corr.fit_transform(df_targets_imputed),
                                                      columns=df_targets_imputed.columns)

                valid_targets_for_plot = [t for t in eda_target_cols if t in df_targets_imputed.columns]
                valid_features_for_plot = [f for f in features_for_corr if f in df_targets_imputed.columns]

                if valid_targets_for_plot and valid_features_for_plot and not df_targets_imputed.empty:
                    target_corr_data = df_targets_imputed.corr()[valid_targets_for_plot].loc[valid_features_for_plot]
                    if not target_corr_data.empty:
                        fig_target_corr = plt.figure(figsize=(
                        max(4, len(valid_targets_for_plot) * 1.5), max(10, len(valid_features_for_plot) / 3.5)))
                        sns.heatmap(target_corr_data.sort_values(by=valid_targets_for_plot[0], ascending=False),
                                    annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
                        plt.title('Feature Correlation with Targets');
                        plt.tight_layout();
                        fig_target_corr.savefig(REPORT_DIR_EDA / "feature_target_corr.png");
                        plt.close(fig_target_corr)

    print("\n--- Outlier Detection (Sample Features) ---")
    if features_for_corr:
        numeric_sample_features = [f for f in features_for_corr if
                                   f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if numeric_sample_features:
            sample_size = min(5, len(numeric_sample_features))
            if sample_size > 0:
                sampled_features_for_boxplot = np.random.choice(numeric_sample_features, sample_size, replace=False)
                fig_outliers = plt.figure(figsize=(3 * sample_size, 5))
                for i, col in enumerate(sampled_features_for_boxplot):
                    plt.subplot(1, sample_size, i + 1)
                    sns.boxplot(y=df[col].dropna())
                    plt.title(col, fontsize=10)
                plt.tight_layout();
                fig_outliers.savefig(REPORT_DIR_EDA / "sample_outliers.png");
                plt.close(fig_outliers)

    print("--- Direct EDA Complete ---")


if __name__ == '__main__':
    run_direct_eda()