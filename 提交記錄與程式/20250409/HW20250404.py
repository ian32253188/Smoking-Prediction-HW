import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier

# Step 1: Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

print("Initial shapes:")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Step 2: Data preprocessing
# Combine train and test for unified processing
test['smoking'] = np.nan
data = pd.concat([train, test], ignore_index=True)
print("Combined data shape:", data.shape)

# Fill missing values
imputer = SimpleImputer(strategy='median')
data.iloc[:, :] = imputer.fit_transform(data)

# Remove whitespace from column names early
data.columns = data.columns.str.replace(' ', '_')

# Identify numerical and categorical columns
categorical_columns = ['hearing(left)', 'hearing(right)', 'Urine_protein', 'dental_caries']
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in ['smoking']]

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)

# Process numerical features first
scaler = MinMaxScaler()
power_transformer = PowerTransformer(method='yeo-johnson')  # yeo-johnson can handle negative values
data[numerical_columns] = power_transformer.fit_transform(data[numerical_columns])
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# OneHot encoding for categorical columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(
    encoded, 
    columns=encoder.get_feature_names_out(categorical_columns),
    index=data.index
)

# Remove original categorical columns and add encoded ones
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_df], axis=1)

# Add KMeans clustering feature last
kmeans = KMeans(n_clusters=5, random_state=42)
data['kmeans_cluster'] = kmeans.fit_predict(data.drop(columns=['smoking']))

print("Features after processing:", data.columns.tolist())

# Get the original lengths of train and test datasets
train_length = len(train)
test_length = len(test)

# Split into train and test using the original lengths
X_train = data.iloc[:train_length].drop(columns=['smoking'])
X_test = data.iloc[train_length:].drop(columns=['smoking'])
y_train = data.iloc[:train_length]['smoking'].astype(int)

print("Final shapes:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)

# Verify data is not empty
if X_test.empty or X_test.shape[1] == 0:
    raise ValueError("X_test is empty or has no features.")

# Step 3: Model training and cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_model = XGBClassifier(tree_method='hist', eval_metric='logloss', use_label_encoder=False, base_score=0.5)
lgbm_model = LGBMClassifier(objective='binary')
catboost_model = CatBoostClassifier(verbose=0)

xgb_preds = np.zeros(len(X_test))
lgbm_preds = np.zeros(len(X_test))
catboost_preds = np.zeros(len(X_test))

for train_idx, val_idx in skf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    try:
        # XGBoost
        xgb_model.fit(X_tr, y_tr)
        xgb_preds += xgb_model.predict_proba(X_test)[:, 1] / skf.n_splits
    except Exception as e:
        print(f"XGBoost training error: {e}")

    try:
        # LightGBM
        lgbm_model.fit(X_tr, y_tr)
        lgbm_preds += lgbm_model.predict_proba(X_test)[:, 1] / skf.n_splits
    except Exception as e:
        print(f"LightGBM training error: {e}")

    try:
        # CatBoost
        catboost_model.fit(X_tr, y_tr)
        catboost_preds += catboost_model.predict_proba(X_test)[:, 1] / skf.n_splits
    except Exception as e:
        print(f"CatBoost training error: {e}")

# Step 4: Model ensembling
final_preds = 0.34 * xgb_preds + 0.33 * lgbm_preds + 0.33 * catboost_preds

# Step 5: Prediction and output
sample_submission['smoking'] = final_preds
sample_submission.to_csv('submission.csv', index=False)