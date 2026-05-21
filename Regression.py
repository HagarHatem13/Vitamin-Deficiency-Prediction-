

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('/content/train_data.csv')
df.head()

#understanding the dataset
df.shape
df.head()
df.info()
df.describe()

#total null value
df.isnull().sum()
#handling null values
#creates flag, 1 if missing, 0 if not missing
df['alcohol_missing'] = df['alcohol_consumption'].isna().astype(int)
df['symptoms_missing'] = df['symptoms_list'].isna().astype(int)

df['alcohol_consumption'] = df['alcohol_consumption'].fillna('Unknown')
df['symptoms_list'] = df['symptoms_list'].fillna('None')

#no duplicates found
duplicates_total = df.duplicated().sum()
print(duplicates_total)

#numerical features analysis

num_features = [
    'age', 'bmi',
    'vitamin_a_percent_rda',
    'vitamin_c_percent_rda',
    'vitamin_d_percent_rda',
    'vitamin_e_percent_rda',
    'vitamin_b12_percent_rda',
    'folate_percent_rda',
    'calcium_percent_rda',
    'iron_percent_rda',
    'symptoms_count',
    'vitamin_deficiency'
]

#show distrbution of values using histogram
df[num_features].hist(figsize=(15, 12), bins=20)
plt.tight_layout()
plt.show()


#show boxplots to identify outliers
plt.figure(figsize=(15, 8))
for i, col in enumerate(num_features[:-1]):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()


#scatter plots vs target
target = 'vitamin_deficiency'


features_for_scatter = num_features[:-1]
num_plots = len(features_for_scatter)


rows = (num_plots + 3) // 4
cols = 4

plt.figure(figsize=(cols * 4, rows * 3))
for i, col in enumerate(features_for_scatter):
    plt.subplot(rows, cols, i + 1)
    sns.scatterplot(x=df[col], y=df[target])
    plt.title(f"{col} vs {target}")

plt.tight_layout()
plt.show()


#correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_features].corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap")
plt.show()

#categorical features analysis

cat_features = [
    'gender',
    'smoking_status',
    'alcohol_consumption',
    'exercise_level',
    'diet_type',
    'sun_exposure',
    'income_level',
    'latitude_region'
]


num_cat_features = len(cat_features)
rows = (num_cat_features + 3) // 4
cols = 4

#count plots to show categorical features distributions
plt.figure(figsize=(cols * 5, rows * 4))
plt.suptitle('Distribution of Categorical Features', fontsize=16, y=1.02)
for i, col in enumerate(cat_features):
    plt.subplot(rows, cols, i + 1)
    sns.countplot(x=df[col])
    plt.title(f"Distribution of {col}", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('')
plt.tight_layout()
plt.show()

#target = vitamin deficiency
#box plots vs target
plt.figure(figsize=(cols * 5, rows * 4))
plt.suptitle('Categorical Features vs Vitamin Deficiency (Box Plots)', fontsize=16, y=1.02)
for i, col in enumerate(cat_features):
    plt.subplot(rows, cols, i + 1)
    sns.boxplot(x=df[col], y=df['vitamin_deficiency'])
    plt.title(f"{col} vs vitamin_deficiency", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.show()

#mean comparison with bar plots with target
plt.figure(figsize=(cols * 5, rows * 4))
plt.suptitle('Mean Vitamin Deficiency by Categorical Features (Bar Plots)', fontsize=16, y=1.02)
for i, col in enumerate(cat_features):
    plt.subplot(rows, cols, i + 1)
    sns.barplot(x=df[col], y=df['vitamin_deficiency'])
    plt.title(f"Mean deficiency by {col}", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.show()

#additional feature vs feature analysis

plt.figure(figsize=(16, 10))


#bmi vs exercise level
plt.subplot(2, 2, 1)
sns.boxplot(x='exercise_level', y='bmi', data=df)
plt.title("BMI vs Exercise Level")
plt.xticks(rotation=30)


#vitamin d vs sun exposure

plt.subplot(2, 2, 2)
sns.boxplot(x='sun_exposure', y='vitamin_d_percent_rda', data=df)
plt.title("Vitamin D vs Sun Exposure")
plt.xticks(rotation=30)


#vitamin b12 vs diet type
plt.subplot(2, 2, 3)
sns.boxplot(x='diet_type', y='vitamin_b12_percent_rda', data=df)
plt.title("Vitamin B12 vs Diet Type")
plt.xticks(rotation=30)


#vitamin deficiency vs income level
plt.subplot(2, 2, 4)
sns.boxplot(x='income_level', y='diet_type', data=df)
plt.title("Diet Type vs Income Level")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

#feature engineering



vitamin_cols = [
    'vitamin_a_percent_rda',
    'vitamin_c_percent_rda',
    'vitamin_d_percent_rda',
    'vitamin_e_percent_rda',
    'vitamin_b12_percent_rda',
    'folate_percent_rda',
    'calcium_percent_rda',
    'iron_percent_rda'
]

df['total_vitamin_intake'] = df[vitamin_cols].sum(axis=1)



df['vitamin_gap'] = (100 - df[vitamin_cols]).clip(lower=0).sum(axis=1)

#feature engineering features analysis

plt.figure(figsize=(18, 12))
plt.suptitle('Feature Engineering Analysis', fontsize=16)

#total vitamin intake distribution
plt.subplot(3, 2, 1)
sns.histplot(df['total_vitamin_intake'], bins=30, kde=True)
plt.title('Distribution of Total Vitamin Intake')

#total vitamin intake vs vitamin deficiency
plt.subplot(3, 2, 2)
sns.scatterplot(x=df['total_vitamin_intake'], y=df['vitamin_deficiency'])
plt.title('Total Vitamin Intake vs Vitamin Deficiency')

#vitamin gap distribution
plt.subplot(3, 2, 3)
sns.histplot(df['vitamin_gap'], bins=30, kde=True)
plt.title('Distribution of Vitamin Gap (Deficit Score)')

#vitamin gap vs vitamin deficiency
plt.subplot(3, 2, 4)
sns.scatterplot(x=df['vitamin_gap'], y=df['vitamin_deficiency'])
plt.title('Vitamin Gap vs Vitamin Deficiency')

#correlation heatmap
plt.subplot(3, 2, (5, 6))
sns.heatmap(df[['total_vitamin_intake', 'vitamin_gap', 'vitamin_deficiency']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Engineered Features')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

df.info()

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

cat_features = [
    'gender','smoking_status','alcohol_consumption',
    'exercise_level','diet_type','sun_exposure',
    'income_level','latitude_region'
]

encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

encoded_cat = encoder.fit_transform(df[cat_features])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_features),index=df.index)

# Combine with numerical
df_encoded = pd.concat([df.drop(columns=cat_features), encoded_cat_df], axis=1)

df.head()

plt.figure(figsize=(18, 12))
sns.heatmap(
    df_encoded.drop(columns=['symptoms_list'], errors='ignore').corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Heatmap (Encoded Dataset)")
plt.show()

#feature selection using high correlation
correlation_with_target = df_encoded.drop(columns=['symptoms_list'], errors='ignore').corr()['vitamin_deficiency'].abs()
high_corr_features = correlation_with_target[correlation_with_target >= 0.3].index.tolist()


if 'vitamin_deficiency' not in high_corr_features:
    high_corr_features.append('vitamin_deficiency')


df_high_corr = df_encoded[high_corr_features]

print("Features with absolute correlation >= 0.3 with vitamin_deficiency:")
print(high_corr_features)
print("\nShape of the new dataset:", df_high_corr.shape)
print("\nHead of the new dataset:")
print(df_high_corr.head())

!pip install catboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

#high correlation dataset modeling
def plot_regression_results(y_test, y_pred, model_name, ax):
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)


    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model_name} - Actual vs Predicted")


X_high_corr = df_high_corr.drop('vitamin_deficiency', axis=1)
y_high_corr = df_high_corr['vitamin_deficiency']


X_train_high_corr, X_test_high_corr, y_train_high_corr, y_test_high_corr = train_test_split(
    X_high_corr, y_high_corr,
    test_size=0.2,
    random_state=42
)


scaler_high_corr = StandardScaler()
X_train_scaled_high_corr = scaler_high_corr.fit_transform(X_train_high_corr)
X_test_scaled_high_corr = scaler_high_corr.transform(X_test_high_corr)

print("\n--- Linear Regression on High Correlation Features ---")
lr_high_corr = LinearRegression()
lr_high_corr.fit(X_train_scaled_high_corr, y_train_high_corr)
pred_lr_high_corr = lr_high_corr.predict(X_test_scaled_high_corr)
rmse_lr_high_corr = np.sqrt(mean_squared_error(y_test_high_corr, pred_lr_high_corr))
r2_lr_high_corr = r2_score(y_test_high_corr, pred_lr_high_corr)
print(f"Linear Regression (High Corr) RMSE: {rmse_lr_high_corr:.4f}")
print(f"Linear Regression (High Corr) R2: {r2_lr_high_corr:.4f}")

print("\n--- Random Forest Regressor on High Correlation Features ---")
rf_high_corr = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_high_corr.fit(X_train_high_corr, y_train_high_corr)
pred_rf_high_corr = rf_high_corr.predict(X_test_high_corr)
rmse_rf_high_corr = np.sqrt(mean_squared_error(y_test_high_corr, pred_rf_high_corr))
r2_rf_high_corr = r2_score(y_test_high_corr, pred_rf_high_corr)
print(f"Random Forest (High Corr) RMSE: {rmse_rf_high_corr:.4f}")
print(f"Random Forest (High Corr) R2: {r2_rf_high_corr:.4f}")


print("\n--- CatBoost Regressor on High Correlation Features ---")
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=0,
    random_state=42
)

cat_model.fit(X_train_high_corr, y_train_high_corr,
    eval_set=(X_test_high_corr, y_test_high_corr),
    early_stopping_rounds=50)

pred_cat = cat_model.predict(X_test_high_corr)
rmse_cat = np.sqrt(mean_squared_error(y_test_high_corr, pred_cat))
r2_cat = r2_score(y_test_high_corr, pred_cat)
print(f"CatBoost RMSE: {rmse_cat:.4f}")
print(f"CatBoost R2: {r2_cat:.4f}")

print("\n--- Model Comparison on High Correlation Features ---")
print(f"Linear Regression RMSE: {rmse_lr_high_corr:.4f} | R2: {r2_lr_high_corr:.4f}")
print(f"Random Forest RMSE: {rmse_rf_high_corr:.4f} | R2: {r2_rf_high_corr:.4f}")
print(f"CatBoost RMSE: {rmse_cat:.4f} | R2: {r2_cat:.4f}")



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_regression_results(y_test_high_corr, pred_lr_high_corr, "Linear Regression", axes[0])
plot_regression_results(y_test_high_corr, pred_rf_high_corr, "Random Forest", axes[1])
plot_regression_results(y_test_high_corr, pred_cat, "CatBoost", axes[2])

plt.tight_layout()
plt.show()

#feature selection using random forest importance and modeling

X = df_encoded.drop(columns=['vitamin_deficiency', 'symptoms_list'], errors='ignore')
y = df_encoded['vitamin_deficiency']

rf_fs = RandomForestRegressor(n_estimators=300, random_state=42)
rf_fs.fit(X, y)

importances = pd.Series(rf_fs.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

top_k = 12
top_features = importances.head(top_k).index

X_selected = X[top_features]

print("Selected Features:\n", list(top_features))


X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)


rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)


cat = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=0,
    random_state=42
)
cat.fit(X_train, y_train, eval_set=(X_test, y_test),
    early_stopping_rounds=50)
pred_cat = cat.predict(X_test)


print("\nMODEL COMPARISON")

print("Linear RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("RF RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))
print("CatBoost RMSE:", np.sqrt(mean_squared_error(y_test, pred_cat)))

print("\nR2 Scores:")
print("Linear:", r2_score(y_test, pred_lr))
print("RF:", r2_score(y_test, pred_rf))
print("CatBoost:", r2_score(y_test, pred_cat))



fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_regression_results(y_test, pred_lr, "Linear Regression", axes[0])
plot_regression_results(y_test, pred_rf, "Random Forest", axes[1])
plot_regression_results(y_test, pred_cat, "CatBoost", axes[2])

plt.tight_layout()
plt.show()

import pickle

# =========================
# SAVE FEATURE SELECTION
# =========================
high_corr_features_without_target = [
    col for col in high_corr_features
    if col != 'vitamin_deficiency'
]

with open("high_corr_features.pkl", "wb") as f:
    pickle.dump(high_corr_features_without_target, f)

with open("top_features.pkl", "wb") as f:
    pickle.dump(top_features, f)

# =========================
# SAVE ENCODING STRUCTURE
# =========================
# Save column names after one-hot encoding
with open("encoded_columns.pkl", "wb") as f:
    pickle.dump(df_encoded.columns.tolist(), f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# =========================
# SAVE SCALERS
# =========================
with open("scaler_high_corr.pkl", "wb") as f:
    pickle.dump(scaler_high_corr, f)

with open("scaler_feature_selection.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =========================
# SAVE MODELS (HIGH CORR)
# =========================
with open("lr_high_corr.pkl", "wb") as f:
    pickle.dump(lr_high_corr, f)

with open("rf_high_corr.pkl", "wb") as f:
    pickle.dump(rf_high_corr, f)

with open("cat_high_corr.pkl", "wb") as f:
    pickle.dump(cat_model, f)

# =========================
# SAVE MODELS (FEATURE IMPORTANCE)
# =========================
with open("lr_fs.pkl", "wb") as f:
    pickle.dump(lr, f)

with open("rf_fs.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("cat_fs.pkl", "wb") as f:
    pickle.dump(cat, f)

print("\nâ All models, scalers, and preprocessing objects saved successfully!")

# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import mean_squared_error, r2_score

# =========================
# LOAD TEST DATA
# =========================
df_test = pd.read_csv('/content/test_data.csv')

y_test = df_test['vitamin_deficiency']
X_test = df_test.drop(columns=['vitamin_deficiency'], errors='ignore')

# =========================
# LOAD SAVED OBJECTS
# =========================
encoder = pickle.load(open("encoder.pkl", "rb"))
trained_columns = pickle.load(open("encoded_columns.pkl", "rb"))

high_corr_features = pickle.load(open("high_corr_features.pkl", "rb"))
top_features = pickle.load(open("top_features.pkl", "rb"))

scaler_high_corr = pickle.load(open("scaler_high_corr.pkl", "rb"))
scaler_fs = pickle.load(open("scaler_feature_selection.pkl", "rb"))

# Models
lr_high_corr = pickle.load(open("lr_high_corr.pkl", "rb"))
rf_high_corr = pickle.load(open("rf_high_corr.pkl", "rb"))
cat_high_corr = pickle.load(open("cat_high_corr.pkl", "rb"))

lr_fs = pickle.load(open("lr_fs.pkl", "rb"))
rf_fs = pickle.load(open("rf_fs.pkl", "rb"))
cat_fs = pickle.load(open("cat_fs.pkl", "rb"))

# =========================
# MISSING VALUES HANDLING
# =========================
X_test['alcohol_missing'] = X_test['alcohol_consumption'].isna().astype(int)
X_test['symptoms_missing'] = X_test['symptoms_list'].isna().astype(int)

X_test['alcohol_consumption'] = X_test['alcohol_consumption'].fillna('Unknown')
X_test['symptoms_list'] = X_test['symptoms_list'].fillna('None')

# =========================
# FEATURE ENGINEERING
# =========================
vitamin_cols = [
    'vitamin_a_percent_rda',
    'vitamin_c_percent_rda',
    'vitamin_d_percent_rda',
    'vitamin_e_percent_rda',
    'vitamin_b12_percent_rda',
    'folate_percent_rda',
    'calcium_percent_rda',
    'iron_percent_rda'
]

X_test['total_vitamin_intake'] = X_test[vitamin_cols].sum(axis=1)
X_test['vitamin_gap'] = (100 - X_test[vitamin_cols]).clip(lower=0).sum(axis=1)

# =========================
# ONE HOT ENCODING
# =========================
cat_features = [
    'gender','smoking_status','alcohol_consumption',
    'exercise_level','diet_type','sun_exposure',
    'income_level','latitude_region'
]

encoded_array = encoder.transform(X_test[cat_features])

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(cat_features),
    index=X_test.index
)

X_test_encoded = pd.concat(
    [X_test.drop(columns=cat_features), encoded_df],
    axis=1
)

# =========================
# ALIGN WITH TRAINING
# =========================
X_test_encoded = X_test_encoded.reindex(columns=trained_columns, fill_value=0)

# =========================
# PIPELINE 1: HIGH CORR (DEPLOYMENT ONLY)
# =========================
X_high = X_test_encoded[high_corr_features]
X_high_scaled = scaler_high_corr.transform(X_high)

pred_lr_high = lr_high_corr.predict(X_high_scaled)
pred_rf_high = rf_high_corr.predict(X_high)
pred_cat_high = cat_high_corr.predict(X_high)

# =========================
# PIPELINE 2: FEATURE SELECTION (DEPLOYMENT ONLY)
# =========================
X_fs = X_test_encoded[top_features]
X_fs_scaled = scaler_fs.transform(X_fs)

pred_lr_fs = lr_fs.predict(X_fs_scaled)
pred_rf_fs = rf_fs.predict(X_fs)
pred_cat_fs = cat_fs.predict(X_fs)

# =========================
# EVALUATION (ONLY BECAUSE LABELS EXIST)
# =========================
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

print("\n--- HIGH CORRELATION PIPELINE ---")
evaluate(y_test, pred_lr_high, "LR")
evaluate(y_test, pred_rf_high, "RF")
evaluate(y_test, pred_cat_high, "CatBoost")

print("\n--- FEATURE SELECTION PIPELINE ---")
evaluate(y_test, pred_lr_fs, "LR")
evaluate(y_test, pred_rf_fs, "RF")
evaluate(y_test, pred_cat_fs, "CatBoost")
