
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('linear_regression_3.csv')

y = df[['y']].copy()           
X = df.drop(columns=['y']).copy()  

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Add constant 
X_train_const = sm.add_constant(X_train_scaled)
full_model = sm.OLS(y_train, X_train_const).fit()
print("Full model summary (using all features):")
print(full_model.summary())

# --- Single-feature simple linear regressions ---
print("\nSingle-feature regressions (train splits):")
for col in X.columns:
    Xf_train = X_train[[col]].copy()
    Xf_test = X_test[[col]].copy()

    # scale single feature
    scaler_f = StandardScaler()
    Xf_train_scaled = pd.DataFrame(scaler_f.fit_transform(Xf_train), columns=[col], index=Xf_train.index)
    Xf_train_const = sm.add_constant(Xf_train_scaled)

    model_f = sm.OLS(y_train, Xf_train_const).fit()
    r2 = model_f.rsquared
    pvals = model_f.pvalues 
    print(f"\nFeature: {col}")
    print(f"  R-squared (train): {r2:.4f}")
    print(f"  p-value (coef): {pvals.get(col):.4g}")

# vif
X_vif = sm.add_constant(X_train_scaled)
vif_data = []
for i, feature in enumerate(X_vif.columns):
    if feature == "const":
        continue
    vif_value = variance_inflation_factor(X_vif.values, X_vif.columns.get_loc(feature))
    vif_data.append((feature, vif_value))

vif_df = pd.DataFrame(vif_data, columns=["feature", "VIF"]).sort_values("VIF", ascending=False).reset_index(drop=True)
print("\nVIF (calculated on scaled training features):")
print(vif_df)

# droppig features
VIF_THRESHOLD = 10.0
to_drop = vif_df[vif_df["VIF"] > VIF_THRESHOLD]["feature"].tolist()
if to_drop:
    print(f"\nDropping features with VIF > {VIF_THRESHOLD}: {to_drop}")
else:
    print(f"\nNo features exceed VIF > {VIF_THRESHOLD}")

# drop the features
X_reduced = X.drop(columns=to_drop) if to_drop else X.copy()

# after dropping
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
scaler_r = StandardScaler()
Xr_train_scaled = pd.DataFrame(scaler_r.fit_transform(Xr_train), columns=Xr_train.columns, index=Xr_train.index)
Xr_test_scaled = pd.DataFrame(scaler_r.transform(Xr_test), columns=Xr_test.columns, index=Xr_test.index)

Xr_train_const = sm.add_constant(Xr_train_scaled)
reduced_model = sm.OLS(yr_train, Xr_train_const).fit()
print("\nReduced model summary (after dropping high-VIF features):")
print(reduced_model.summary())

resid = reduced_model.resid

