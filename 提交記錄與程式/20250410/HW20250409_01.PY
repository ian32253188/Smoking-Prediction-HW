import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from datetime import datetime
import optuna
import shap
import time
import warnings

warnings.filterwarnings("ignore")
start_time = time.time()

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

test['smoking'] = np.nan
data = pd.concat([train, test], ignore_index=True)
data.columns = data.columns.str.replace(' ', '_')

# Define columns
categorical_columns = ['hearing(left)', 'hearing(right)', 'Urine_protein', 'dental_caries']
categorical_columns = [col.replace(' ', '_') for col in categorical_columns]
numerical_columns = [col for col in data.columns if col not in categorical_columns + ['smoking']
                     and data[col].dtype in ['float64', 'int64']]

# Imputation & Normalization
imputer = SimpleImputer(strategy='median')
data[data.columns] = imputer.fit_transform(data[data.columns])
power_transformer = PowerTransformer(method='yeo-johnson')
scaler = MinMaxScaler()
data[numerical_columns] = power_transformer.fit_transform(data[numerical_columns])
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# One-hot encode
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns), index=data.index)
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_df], axis=1)

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(data[numerical_columns])

# Split
train_length = len(train)
X = data.iloc[:train_length].drop(columns=['smoking'])
X_test = data.iloc[train_length:].drop(columns=['smoking'])
y = data.iloc[:train_length]['smoking'].astype(int)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
features = X.columns

# === XGBoost ===
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
        'device': 'cuda',
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    trial.set_user_attr("params", params)
    trial.set_user_attr("mean_auc", auc)
    return auc

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)
best_xgb = xgb.XGBClassifier(**study_xgb.best_params, tree_method='hist', device='cuda', eval_metric='logloss')
best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_log = best_xgb.evals_result()
xgb_preds = best_xgb.predict_proba(X_test)[:, 1]

# === LightGBM ===
def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'device_type': 'gpu',
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
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30)])
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    trial.set_user_attr("params", params)
    trial.set_user_attr("mean_auc", auc)
    return auc

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)
best_lgb = LGBMClassifier(**study_lgb.best_params, verbosity=-1)
best_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(30)])
lgb_log = best_lgb.evals_result_
lgb_preds = best_lgb.predict_proba(X_test)[:, 1]

# === CatBoost ===
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
        'task_type': 'GPU',
        'od_type': 'Iter',
        'od_wait': 30
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    trial.set_user_attr("params", params)
    trial.set_user_attr("mean_auc", auc)
    return auc

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)
best_cat = CatBoostClassifier(**study_cat.best_params)
best_cat.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
cat_log = best_cat.get_evals_result()
cat_preds = best_cat.predict_proba(X_test)[:, 1]

# Auto Weighted Ensemble
auc_xgb = study_xgb.best_value or 0
auc_lgb = study_lgb.best_value or 0
auc_cat = study_cat.best_value or 0
total_auc = auc_xgb + auc_lgb + auc_cat
w_xgb = auc_xgb / total_auc
w_lgb = auc_lgb / total_auc
w_cat = auc_cat / total_auc
print(f"✅ 使用自動 AUC 權重：XGB = {w_xgb:.4f}, LGB = {w_lgb:.4f}, CAT = {w_cat:.4f}")
final_preds = w_xgb * xgb_preds + w_lgb * lgb_preds + w_cat * cat_preds

# Save prediction
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_{timestamp}.csv'
sample_submission['smoking'] = final_preds
sample_submission.to_csv(filename, index=False)
print(f"📄 Submission saved to {filename}")

# Save Optuna trials
optuna_trials = []
for study, name in [(study_xgb, 'XGBoost'), (study_lgb, 'LightGBM'), (study_cat, 'CatBoost')]:
    for t in study.trials:
        optuna_trials.append({
            'model': name,
            'score': t.user_attrs.get('mean_auc', t.value),
            **t.user_attrs.get('params', {})
        })
pd.DataFrame(optuna_trials).to_csv('optuna_trials_log.csv', index=False)

# Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(xgb_log['validation_0']['logloss'], label='XGBoost Logloss')
plt.plot(lgb_log['valid_0']['auc'], label='LightGBM AUC')
plt.plot(cat_log['validation']['AUC'], label='CatBoost AUC')
plt.xlabel('Iterations')
plt.ylabel('Metric')
plt.title('Training Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()

# Feature Importances
def plot_importance(importances, title, filename):
    df = pd.DataFrame({'Feature': features, 'Importance': importances})
    df = df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(df['Feature'][:20][::-1], df['Importance'][:20][::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_importance(best_xgb.feature_importances_, "XGBoost Feature Importance", "feature_importance_xgb.png")
plot_importance(best_lgb.feature_importances_, "LightGBM Feature Importance", "feature_importance_lgb.png")
plot_importance(best_cat.get_feature_importance(), "CatBoost Feature Importance", "feature_importance_cat.png")

# SHAP Summary Plot for XGBoost
explainer = shap.Explainer(best_xgb, X_train)
shap_values = explainer(X_val)
shap.summary_plot(shap_values, X_val, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png")
plt.show()

print("✅ 所有輸出已完成")
print(f"⏱️ Total runtime: {time.time() - start_time:.2f} seconds")
