# %% [markdown]
# # 資料挖掘作業 2：吸菸預測模型
# 
# 本專案透過多個機器學習模型的集成方法來預測吸菸狀態。我們使用了三種強大的梯度提升樹模型：XGBoost、LightGBM 和 CatBoost，並透過 Optuna 進行超參數優化，最後根據驗證集上的 AUC 得分進行加權集成。

# %% [markdown]
# ## 1. 導入所需套件
# 
# 首先導入所有需要使用的套件和函式庫。

# %%
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

# %% [markdown]
# ## 2. 資料載入與探索
# 
# 在這個部分，我們載入訓練和測試資料集，並進行初步的資料探索。

# %%
# 載入資料集
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# 顯示訓練集的基本資訊
print(f"訓練集形狀: {train.shape}")
print(f"測試集形狀: {test.shape}")
train.head()

# %%
# 檢查訓練集和測試集的缺失值
print("訓練集缺失值統計:")
missing_train = train.isnull().sum()
missing_percentage_train = 100 * missing_train / len(train)
missing_info_train = pd.concat([missing_train, missing_percentage_train], axis=1)
missing_info_train.columns = ['缺失值數量', '缺失百分比 (%)']
missing_info_train = missing_info_train[missing_info_train['缺失值數量'] > 0]
print(missing_info_train if not missing_info_train.empty else "訓練集無缺失值")

print("\n測試集缺失值統計:")
missing_test = test.isnull().sum()
missing_percentage_test = 100 * missing_test / len(test)
missing_info_test = pd.concat([missing_test, missing_percentage_test], axis=1)
missing_info_test.columns = ['缺失值數量', '缺失百分比 (%)']
missing_info_test = missing_info_test[missing_info_test['缺失值數量'] > 0]
print(missing_info_test if not missing_info_test.empty else "測試集無缺失值")

# %% [markdown]
# ## 3. 資料前處理
# 
# 資料前處理是機器學習流程中非常重要的一環。在這一部分，我們將執行以下步驟：
# 1. 合併訓練集和測試集以進行一致的特徵工程
# 2. 處理欄位名稱（替換空格為下劃線）
# 3. 辨識並分類特徵（類別型和數值型）
# 4. 進行缺失值處理和特徵轉換

# %%
# 合併訓練集和測試集以統一進行特徵處理
test['smoking'] = np.nan
data = pd.concat([train, test], ignore_index=True)

# 3. Feature Engineering

data['BMI'] = data['weight(kg)'] / (data['height(cm)'] / 100) ** 2
# data['LDL_to_HDL'] = data['LDL'] / (data['HDL'] + 1e-5)
# data['waist_BMI_ratio'] = data['waist(cm)'] / data['BMI']
# data['liver_mean'] = data[['AST', 'ALT', 'Gtp']].mean(axis=1)

# 定義類別型和數值型特徵
categorical_columns = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']
numerical_columns = [col for col in data.columns if col not in categorical_columns + ['smoking', 'id']
                     and data[col].dtype in ['float64', 'int64']]

# 顯示分類後的特徵數量
print(f"類別型特徵數量: {len(categorical_columns)}")
print(f"數值型特徵數量: {len(numerical_columns)}")

# %% [markdown]
# ### 3.1 缺失值處理與特徵轉換
# 
# 我們使用以下方法處理資料：
# 1. 使用 SimpleImputer 填補缺失值（以中位數填補）
# 2. 應用 Yeo-Johnson 變換來處理偏態分佈
# 3. 使用 MinMaxScaler 將數值特徵縮放到相同範圍

# %%

# 應用 Power Transform 處理偏態分佈
power_transformer = PowerTransformer(method='yeo-johnson')
data[numerical_columns] = power_transformer.fit_transform(data[numerical_columns])

# 特徵縮放
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# %% [markdown]
# ### 3.2 類別型特徵編碼與特徵工程
# 
# 在這個部分，我們將：
# 1. 對類別型特徵進行 One-Hot 編碼
# 2. 應用 KMeans 聚類作為特徵工程的一部分，創建新的聚類特徵

# %%
# One-Hot 編碼
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns), index=data.index)
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_df], axis=1)

# %%

# 使用 KMeans 進行聚類特徵工程
kmeans = KMeans(n_clusters=5, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(data[numerical_columns])

# 檢視特徵工程後的資料
print(f"處理後的特徵數量: {data.shape[1]}")

# %%
print(data.columns)

# %% [markdown]
# ## 4. 資料分割
# 
# 將資料分割為訓練集、驗證集和測試集。這是模型訓練和評估的重要步驟。

# %%
# 定義需要省略的特徵
features_to_drop = ['hearing(left)_2.0', 'hearing(right)_2.0']  # 替換為你想刪除的特徵名稱

# 合併訓練集和測試集後，刪除不需要的特徵
data = data.drop(columns=features_to_drop, errors='ignore')

# 分割資料
train_length = len(train)
X = data.iloc[:train_length].drop(columns=['smoking', 'id'], errors='ignore')
X_test = data.iloc[train_length:].drop(columns=['smoking', 'id'], errors='ignore')
y = data.iloc[:train_length]['smoking'].astype(int)

# 分割訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# 檢查刪除後的數據集形狀
print(f"訓練集形狀: {X_train.shape}")
print(f"驗證集形狀: {X_val.shape}")
print(f"測試集形狀: {X_test.shape}")

# %% [markdown]
# ## 5. 模型超參數優化與訓練
# 
# 在這個部分，我們將使用 Optuna 來為三種不同的梯度提升樹模型尋找最佳的超參數。超參數優化是提高模型性能的關鍵步驟。

# %% [markdown]
# ### 5.1 XGBoost 模型優化與訓練
# 
# XGBoost 是一種高效能的梯度提升樹實現，特別適合結構化/表格式資料。

# %%
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
        'eval_metric': 'auc'
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

# 創建並執行 Optuna 研究對象
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)

# 顯示最佳參數
print("最佳 XGBoost 參數:")
print(study_xgb.best_params)
print(f"最佳 AUC: {study_xgb.best_value:.4f}")

# 使用最佳參數訓練模型
best_xgb = xgb.XGBClassifier(**study_xgb.best_params, tree_method='hist', device='cuda', eval_metric='logloss')
best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_log = best_xgb.evals_result()
xgb_preds = best_xgb.predict_proba(X_test)[:, 1]

# %% [markdown]
# ### 5.2 LightGBM 模型優化與訓練
# 
# LightGBM 是一個高效、低記憶體佔用的梯度提升框架，使用基於直方圖的分割尋找策略，適合大型資料集。

# %%
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

# 創建並執行 Optuna 研究對象
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)

# 顯示最佳參數
print("最佳 LightGBM 參數:")
print(study_lgb.best_params)
print(f"最佳 AUC: {study_lgb.best_value:.4f}")

# 使用最佳參數訓練模型
best_lgb = LGBMClassifier(**study_lgb.best_params, verbosity=-1)
best_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(30)])
lgb_log = best_lgb.evals_result_
lgb_preds = best_lgb.predict_proba(X_test)[:, 1]

# %% [markdown]
# ### 5.3 CatBoost 模型優化與訓練
# 
# CatBoost 是一種高效能的梯度提升樹實現，尤其擅長處理類別型特徵，並自動處理缺失值。

# %%
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
        'task_type': 'CPU',
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

# 創建並執行 Optuna 研究對象
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=30)

# 顯示最佳參數
print("最佳 CatBoost 參數:")
print(study_cat.best_params)
print(f"最佳 AUC: {study_cat.best_value:.4f}")

# 使用最佳參數訓練模型
best_cat = CatBoostClassifier(**study_cat.best_params)
best_cat.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
cat_log = best_cat.get_evals_result()
cat_preds = best_cat.predict_proba(X_test)[:, 1]

# %% [markdown]
# ## 6. 模型集成與預測
# 
# 在這部分，我們將根據各個模型在驗證集上的 AUC 表現，進行自動加權集成，結合三個模型的預測結果。

# %%
cat_preds

# %%
# 計算基於 AUC 的權重
auc_xgb = study_xgb.best_value or 0
auc_lgb = study_lgb.best_value or 0
auc_cat = study_cat.best_value or 0
total_auc = auc_xgb + auc_lgb + auc_cat
w_xgb = auc_xgb / total_auc
w_lgb = auc_lgb / total_auc
w_cat = auc_cat / total_auc

print(f"使用自動 AUC 權重：XGB = {w_xgb:.4f}, LGB = {w_lgb:.4f}, CAT = {w_cat:.4f}")

# 加權集成預測
final_preds = w_xgb * xgb_preds + w_lgb * lgb_preds + w_cat * cat_preds
#final_preds = xgb_preds 
#final_preds = lgb_preds 
#final_preds = cat_preds
# 生成提交檔案
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_{timestamp}.csv'
sample_submission['smoking'] = final_preds
sample_submission.to_csv(filename, index=False)
print(f"提交檔案已儲存為 {filename}")

# %% [markdown]
# ## 7. 模型評估與分析
# 
# 在這一部分，我們將通過各種方式分析模型性能和特徵重要性。

# %% [markdown]
# ### 7.1 儲存 Optuna 超參數優化結果
# 
# 記錄所有超參數優化試驗的結果，便於後續分析。

# %%
# 收集所有 Optuna 試驗結果
optuna_trials = []
for study, name in [(study_xgb, 'XGBoost'), (study_lgb, 'LightGBM'), (study_cat, 'CatBoost')]:
    for t in study.trials:
        optuna_trials.append({
            'model': name,
            'score': t.user_attrs.get('mean_auc', t.value),
            **t.user_attrs.get('params', {})
        })
pd.DataFrame(optuna_trials).to_csv('optuna_trials_log.csv', index=False)
print("超參數優化記錄已保存至 optuna_trials_log.csv")

# %% [markdown]
# ### 7.2 繪製訓練曲線
# 
# 視覺化三個模型在訓練過程中的性能變化。

# %%
plt.figure(figsize=(10, 6))
plt.plot(xgb_log['validation_0']['logloss'], label='XGBoost Logloss')
plt.plot(lgb_log['valid_0']['auc'], label='LightGBM AUC')
plt.plot(cat_log['validation']['Logloss'], label='CatBoost AUC')
plt.xlabel('Iterations')
plt.ylabel('Evaluation Metric')
plt.title('Model Training Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()
