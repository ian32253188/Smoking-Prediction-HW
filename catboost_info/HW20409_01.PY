import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
import time
import os

start_time = time.time()

# Step 1: Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Backup original
train_orig = train.copy()
test_orig = test.copy()

test['smoking'] = np.nan
data = pd.concat([train, test], ignore_index=True)
data.columns = data.columns.str.replace(' ', '_')

# Fill missing
imputer = SimpleImputer(strategy='median')
data[data.columns] = imputer.fit_transform(data[data.columns])

categorical_columns = ['hearing(left)', 'hearing(right)', 'Urine_protein', 'dental_caries']
categorical_columns = [col.replace(' ', '_') for col in categorical_columns]
numerical_columns = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col != 'smoking']

# Normalize
power_transformer = PowerTransformer(method='yeo-johnson')
scaler = MinMaxScaler()
data[numerical_columns] = power_transformer.fit_transform(data[numerical_columns])
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# One-hot encode
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns), index=data.index)
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_df], axis=1)

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(data.drop(columns=['smoking']))

# Split
train_length = len(train)
X_train = data.iloc[:train_length].drop(columns=['smoking'])
X_test = data.iloc[train_length:].drop(columns=['smoking'])
y_train = data.iloc[:train_length]['smoking'].astype(int)

# ========================
# XGBoost + Log + Plot
# ========================
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
        'tree_method': 'gpu_hist',
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }

    aucs = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        model = XGBClassifier(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                  eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                  verbose=False)
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))

    trial.set_user_attr("params", params)
    trial.set_user_attr("mean_auc", np.mean(aucs))
    return np.mean(aucs)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)

# 最佳模型 + evals_result
best_xgb = XGBClassifier(**study_xgb.best_params)
best_xgb.fit(X_train, y_train,
             eval_set=[(X_train, y_train)],
             eval_metric='logloss',
             verbose=False)

xgb_log = best_xgb.evals_result()
xgb_preds = best_xgb.predict_proba(X_test)[:, 1]

# ========================
# LightGBM + Log + Plot
# ========================
def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device': 'gpu',
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
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                  eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                  verbose=-1)
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))

    trial.set_user_attr("params", params)
    trial.set_user_attr("mean_auc", np.mean(aucs))
    return np.mean(aucs)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)

best_lgb = LGBMClassifier(**study_lgb.best_params)
best_lgb.fit(X_train, y_train,
             eval_set=[(X_train, y_train)],
             eval_metric='auc',
             verbose=False)

lgb_log = best_lgb.evals_result_
lgb_preds = best_lgb.predict_proba(X_test)[:, 1]

# ========================
# CatBoost + Log + Plot
# ========================
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
        'random_state': 42,
        'task_type': 'GPU'
    }

    aucs = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        model = CatBoostClassifier(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                  eval_set=(X_train.iloc[val_idx], y_train.iloc[val_idx]),
                  use_best_model=True)
        preds = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_train.iloc[val_idx], preds))

    trial.set_user_attr("params", params)
    trial.set_user_attr("mean_auc", np.mean(aucs))
    return np.mean(aucs)

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)

best_cat = CatBoostClassifier(**study_cat.best_params)
best_cat.fit(X_train, y_train,
             eval_set=(X_train, y_train),
             use_best_model=True,
             verbose=False)

cat_log = best_cat.get_evals_result()
cat_preds = best_cat.predict_proba(X_test)[:, 1]

# ========================
# Save Optuna trial logs
# ========================
optuna_trials = []

for study, name in [(study_xgb, 'XGBoost'), (study_lgb, 'LightGBM'), (study_cat, 'CatBoost')]:
    for t in study.trials:
        optuna_trials.append({
            'model': name,
            'score': t.user_attrs.get('mean_auc', t.value),
            **t.user_attrs.get('params', {})
        })

pd.DataFrame(optuna_trials).to_csv('optuna_trials_log.csv', index=False)

# ========================
# Plot training curves
# ========================
plt.figure(figsize=(10, 6))
plt.plot(xgb_log['validation_0']['logloss'], label='XGBoost Logloss')
plt.plot(lgb_log['training']['auc'], label='LightGBM AUC')
plt.plot(cat_log['learn']['AUC'], label='CatBoost AUC')
plt.xlabel('Iterations')
plt.ylabel('Metric')
plt.title('Training Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()

# ========================
# Final Ensemble (Auto AUC Weighting) & Output
# ========================

# 從 Optuna 中擷取最佳 AUC 分數作為加權依據
auc_xgb = study_xgb.best_value
auc_lgb = study_lgb.best_value
auc_cat = study_cat.best_value

# Normalize 成權重（加總為 1）
total_auc = auc_xgb + auc_lgb + auc_cat
w_xgb = auc_xgb / total_auc
w_lgb = auc_lgb / total_auc
w_cat = auc_cat / total_auc

print(f"✅ 使用自動 AUC 權重：XGB = {w_xgb:.4f}, LGB = {w_lgb:.4f}, CAT = {w_cat:.4f}")

# 預測加權平均
final_preds = w_xgb * xgb_preds + w_lgb * lgb_preds + w_cat * cat_preds

# 輸出結果
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_{timestamp}.csv'
sample_submission['smoking'] = final_preds
sample_submission.to_csv(filename, index=False)

print(f"📄 Submission saved to {filename}")
print(f"📊 Trial log saved to optuna_trials_log.csv")
print(f"📈 Training curve saved to training_curve.png")
print(f"⏱️ Total runtime: {time.time() - start_time:.2f} seconds")
