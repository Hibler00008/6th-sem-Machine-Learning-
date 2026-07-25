import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

# ---------- user params ----------
target_column = "y"
random_state = 8
cv_folds = 5
# ---------------------------------

df = pd.read_csv("linear_regression_3.csv")
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# keep only numeric columns for regression pipeline (one-hot encode categoricals before this if needed)
X_train = X_train.select_dtypes(include=[np.number]).copy()
X_test = X_test[X_train.columns].copy()  # align columns/order

vif_dropped = []
pval_dropped = []

# ---------- Scaling (fit on train, apply to test) ----------
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ---------- VIF helper (include constant) ----------
def compute_vif(X):
    Xc = sm.add_constant(X)
    vif_vals = []
    # start from 1 to skip 'const'
    for i in range(1, Xc.shape[1]):
        vif_vals.append(variance_inflation_factor(Xc.values, i))
    return pd.DataFrame({"Feature": X.columns, "VIF": vif_vals})

def remove_high_vif_features(X_train_scaled, X_test_scaled, threshold=10):
    X_temp = X_train_scaled.copy()
    X_test_temp = X_test_scaled.copy()
    while X_temp.shape[1] > 0:
        vif = compute_vif(X_temp)
        max_vif = vif["VIF"].max()
        print(f"max_vif : {max_vif}")
        print(vif.sort_values("VIF", ascending=False))
        if max_vif > threshold:
            feature_to_drop = vif.loc[vif["VIF"].idxmax(), "Feature"]
            print(f"dropping feature due to high VIF: {feature_to_drop}")
            X_temp = X_temp.drop(columns=[feature_to_drop])
            X_test_temp = X_test_temp.drop(columns=[feature_to_drop])
            vif_dropped.append(feature_to_drop)
        else:
            break
    return X_temp, X_test_temp

# ---------- CV RMSE helper ----------
def cv_rmse(X, y, cv=cv_folds):
    model = LinearRegression()
    # cross_val_score uses negative MSE when scoring='neg_mean_squared_error'
    scores = cross_val_score(model, X.values, y.values, cv=cv, scoring='neg_mean_squared_error')
    rmse = np.mean(np.sqrt(-scores))
    return rmse

# ---------- p-value removal but with CV check ----------
def remove_insignificant_features_cv(X_train, X_test, y_train, p_value_threshold=0.05, cv=5):
    dropped = []
    X = X_train.copy()
    X_test_local = X_test.copy()
    while True:
        if X.shape[1] == 0:
            break
        Xc = sm.add_constant(X)
        model = sm.OLS(y_train, Xc).fit()
        pvals = model.pvalues.drop("const")
        if pvals.empty:
            break
        max_pval = pvals.max()
        if max_pval <= p_value_threshold:
            break
        candidate = pvals.idxmax()
        # compare CV RMSE before and after dropping
        base_rmse = cv_rmse(X, y_train, cv=cv)
        rmse_dropped = cv_rmse(X.drop(columns=[candidate]), y_train, cv=cv)
        print(f"candidate to drop: {candidate} - pval {max_pval:.4f} | base_rmse {base_rmse:.4f} -> rmse_if_dropped {rmse_dropped:.4f}")
        # allow tiny tolerance
        if rmse_dropped <= base_rmse + 1e-6:
            X = X.drop(columns=[candidate])
            X_test_local = X_test_local.drop(columns=[candidate])
            dropped.append(candidate)
            pval_dropped.append(candidate)
            print(f"dropped {candidate} (improved or equal CV RMSE).")
        else:
            print(f"NOT dropping {candidate} because CV RMSE worsened.")
            break
    # return last fitted statsmodels model as well
    final_model = sm.OLS(y_train, sm.add_constant(X)).fit()
    return X, X_test_local, dropped, final_model

# ---------- Cook's D helper (returns indices; do not auto-remove) ----------
def get_cooks_outliers_indices(X, y, threshold=None):
    if threshold is None:
        threshold = 4 / len(X)
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    outlier_positions = np.where(cooks_d >= threshold)[0]   # positions relative to X.index
    outlier_indices = X.index[outlier_positions].tolist()
    return outlier_indices, cooks_d

# ---------- initial raw model (before cleaning) ----------
X_train_const_raw = sm.add_constant(X_train_scaled)
raw_model = sm.OLS(y_train, X_train_const_raw).fit()
y_train_pred_raw = raw_model.predict(X_train_const_raw)
r2_raw_train = r2_score(y_train, y_train_pred_raw)
print(f"R² Score on Training Set (before cleaning): {r2_raw_train:.4f}")
print(raw_model.summary())

# ---------- VIF removal ----------
X_train_vif_cleaned, X_test_vif_cleaned = remove_high_vif_features(X_train_scaled.copy(), X_test_scaled.copy(), threshold=10)

# ---------- Cook's D detection - do NOT auto remove; evaluate via CV before removing ----------
train_outlier_idxs, cooks_vals = get_cooks_outliers_indices(X_train_vif_cleaned, y_train)
print(f"Cook's D flagged {len(train_outlier_idxs)} training points (threshold 4/n). Indices: {train_outlier_idxs}")

# If you want to test removal, create the no-outlier sets and compare CV:
if len(train_outlier_idxs) > 0:
    X_train_no_out = X_train_vif_cleaned.drop(index=train_outlier_idxs)
    y_train_no_out = y_train.drop(index=train_outlier_idxs)
    rmse_before = cv_rmse(X_train_vif_cleaned, y_train, cv=cv_folds)
    rmse_after = cv_rmse(X_train_no_out, y_train_no_out, cv=cv_folds)
    print(f"CV RMSE before removing Cook's points: {rmse_before:.4f}, after removing: {rmse_after:.4f}")
    # Only if rmse_after <= rmse_before you may choose to remove them
    # If you decide to remove, reassign X_train_vif_cleaned, y_train accordingly.

# For now, proceed WITHOUT auto-removing Cook's points (manual decision recommended)

# ---------- p-value removal with CV checking ----------
X_train_final, X_test_final, dropped_pvals, last_model = remove_insignificant_features_cv(X_train_vif_cleaned.copy(), X_test_vif_cleaned.copy(), y_train, p_value_threshold=0.05, cv=cv_folds)

# ---------- Fit final model on the (possibly same) training set ----------
X_train_const = sm.add_constant(X_train_final)
X_test_const = sm.add_constant(X_test_final)
final_model = sm.OLS(y_train, X_train_const).fit()  # note: if you removed Cook's points earlier, use y_train_no_out

y_train_pred = final_model.predict(X_train_const)
y_test_pred = final_model.predict(X_test_const)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

def adjusted_r2_from_model(model):
    # statsmodels provides rsquared_adj
    return model.rsquared_adj

print("Dropped due to VIF:", vif_dropped)
print("Dropped due to p-value (after CV check):", pval_dropped)
print("Final features used:", X_train_final.columns.tolist())
print(f"R² Score on Training Set (after cleaning): {r2_train:.4f}")
print(f"Adjusted R² (final model): {adjusted_r2_from_model(final_model):.4f}")
print(f"R² Score on Test Set (all points): {r2_test:.4f}")

# ---------- residual plot (test) ----------
def plot_residuals(y_test, y_test_pred):
    residuals = y_test - y_test_pred
    Q1 = np.percentile(residuals, 25)
    Q3 = np.percentile(residuals, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=1, label="Outlier Boundaries")
    plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=1)
    plt.title('Histogram of Test Set Residuals with Outlier Boundaries (IQR)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_residuals(y_test, y_test_pred)

print(f"Final Model Summary:\n{final_model.summary()}")
