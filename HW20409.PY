import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from datetime import datetime
import optuna

# Step 1: Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Always backup original
train_orig = train.copy()
test_orig = test.copy()

# Add placeholder for target in test
test['smoking'] = np.nan

# Combine data
data = pd.concat([train, test], ignore_index=True)

# Clean column names
data.columns = data.columns.str.replace(' ', '_')

# Fill missing values
imputer = SimpleImputer(strategy='median')
data[data.columns] = imputer.fit_transform(data[data.columns])

# Define categorical and numerical columns
categorical_columns = ['hearing(left)', 'hearing(right)', 'Urine_protein', 'dental_caries']
categorical_columns = [col.replace(' ', '_') for col in categorical_columns]
numerical_columns = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col != 'smoking']

# Normalize numerical data
scaler = MinMaxScaler()
power_transformer = PowerTransformer(method='yeo-johnson')
data[numerical_columns] = power_transformer.fit_transform(data[numerical_columns])
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# One-hot encode categorical features
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns), index=data.index)

# Replace original categorical columns
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_df], axis=1)

# Add KMeans feature
kmeans = KMeans(n_clusters=5, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(data.drop(columns=['smoking']))

# Split back
train_length = len(train)
X_train = data.iloc[:train_length].drop(columns=['smoking'])
X_test = data.iloc[train_length:].drop(columns=['smoking'])
y_train = data.iloc[:train_length]['smoking'].astype(int)

# ================================
# Optuna Hyperparameter Tuning
# ================================

# --- XGBoost Tuning ---
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
        'tree_method': 'hist',
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    aucs = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        model = XGBClassifier(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))
    return np.mean(aucs)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)
best_xgb = XGBClassifier(**study_xgb.best_params)
best_xgb.fit(X_train, y_train)
xgb_preds = best_xgb.predict_proba(X_test)[:, 1]

# --- LightGBM Tuning ---
def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }
    aucs = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        model = LGBMClassifier(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))
    return np.mean(aucs)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)
best_lgb = LGBMClassifier(**study_lgb.best_params)
best_lgb.fit(X_train, y_train)
lgb_preds = best_lgb.predict_proba(X_test)[:, 1]

# --- CatBoost Tuning ---
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0,
        'eval_metric': 'AUC',
        'random_state': 42
    }
    aucs = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        model = CatBoostClassifier(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))
    return np.mean(aucs)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)
best_cat = CatBoostClassifier(**study_cat.best_params)
best_cat.fit(X_train, y_train)
cat_preds = best_cat.predict_proba(X_test)[:, 1]

# ================================
# Ensemble and Submission
# ================================
final_preds = 0.34 * xgb_preds + 0.33 * lgb_preds + 0.33 * cat_preds

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_{timestamp}.csv'
sample_submission['smoking'] = final_preds
sample_submission.to_csv(filename, index=False)

print(f"Submission saved to {filename}")
