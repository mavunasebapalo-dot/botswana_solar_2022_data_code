import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("botswana_solar_2022.csv")
feat_cols = ["Latitude", "Longitude", "Total_Households", "Urban"]
Xm = df[feat_cols].values
y_count = df["Solar_Cooking_Count"].values
total_hh = df["Total_Households"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def report(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name}: R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    return r2, rmse, mae

# ---------- Check 4: naive population-scaling baseline (mean rate x population), out-of-fold ----------
print("=" * 70)
print("CHECK 4: Naive baseline (training-fold mean adoption rate x district population)")
print("=" * 70)
naive_pred = np.zeros_like(y_count, dtype=float)
rate = df["Solar_Cooking_Pct"].values / 100.0  # convert % to fraction
for train_idx, test_idx in kf.split(Xm):
    mean_rate_train = rate[train_idx].mean()
    naive_pred[test_idx] = mean_rate_train * total_hh[test_idx]
naive_metrics = report("Naive (mean rate x population)", y_count, naive_pred)

# also refit RF and Linear on count target (5-fold) for side-by-side comparison in this script's output
rf = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=2, random_state=42)
lin = LinearRegression()
rf_pred = cross_val_predict(rf, Xm, y_count, cv=kf)
lin_pred = cross_val_predict(lin, Xm, y_count, cv=kf)
rf_metrics = report("Random Forest (5-fold, count)", y_count, rf_pred)
lin_metrics = report("Linear Regression (5-fold, count)", y_count, lin_pred)

# ---------- Check 5: Gradient Boosting as a second tree-ensemble method ----------
print()
print("=" * 70)
print("CHECK 5: Gradient Boosting Regressor (second tree-ensemble method), count target")
print("=" * 70)
gbr = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
gbr_pred = cross_val_predict(gbr, Xm, y_count, cv=kf)
gbr_metrics = report("Gradient Boosting (5-fold, count)", y_count, gbr_pred)

# ---------- Check 6: paired bootstrap CI for the RMSE/R2 gap (Linear - RF), using existing out-of-fold predictions ----------
print()
print("=" * 70)
print("CHECK 6: Paired bootstrap (10,000 resamples) for Linear vs. RF gap, count target")
print("=" * 70)
rng = np.random.RandomState(42)
n = len(y_count)
B = 10000
rmse_diff = np.zeros(B)   # RF_RMSE - Linear_RMSE  (positive = RF worse)
r2_diff = np.zeros(B)     # Linear_R2 - RF_R2       (positive = Linear better)
mae_diff = np.zeros(B)

for b in range(B):
    idx = rng.randint(0, n, size=n)
    yt = y_count[idx]
    yr = rf_pred[idx]
    yl = lin_pred[idx]
    rmse_rf_b = np.sqrt(mean_squared_error(yt, yr))
    rmse_lin_b = np.sqrt(mean_squared_error(yt, yl))
    rmse_diff[b] = rmse_rf_b - rmse_lin_b
    # guard against degenerate resamples with zero variance in yt
    if np.var(yt) > 0:
        r2_diff[b] = r2_score(yt, yl) - r2_score(yt, yr)
    else:
        r2_diff[b] = np.nan
    mae_diff[b] = mean_absolute_error(yt, yr) - mean_absolute_error(yt, yl)

r2_diff = r2_diff[~np.isnan(r2_diff)]
rmse_ci = np.percentile(rmse_diff, [2.5, 97.5])
r2_ci = np.percentile(r2_diff, [2.5, 97.5])
mae_ci = np.percentile(mae_diff, [2.5, 97.5])
pct_rf_worse_rmse = (rmse_diff > 0).mean() * 100
pct_linear_better_r2 = (r2_diff > 0).mean() * 100

print(f"Observed: RMSE(RF) - RMSE(Linear) = {rf_metrics[1] - lin_metrics[1]:.2f}")
print(f"Bootstrap 95% CI for RMSE(RF) - RMSE(Linear): [{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}]")
print(f"Bootstrap: RF has higher RMSE than Linear in {pct_rf_worse_rmse:.1f}% of resamples")
print(f"Observed: R2(Linear) - R2(RF) = {lin_metrics[0] - rf_metrics[0]:.3f}")
print(f"Bootstrap 95% CI for R2(Linear) - R2(RF): [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]")
print(f"Bootstrap: Linear has higher R2 than RF in {pct_linear_better_r2:.1f}% of resamples")
print(f"Bootstrap 95% CI for MAE(RF) - MAE(Linear): [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]")

# ---------- Check 7: same bootstrap test, but excluding Kweneng East to test whether the whole gap is one district ----------
print()
print("=" * 70)
print("CHECK 7: Drop Kweneng East and recompute all three models (count target, 5-fold)")
print("=" * 70)
mask = df["District"] != "Kweneng East"
Xm2 = df.loc[mask, feat_cols].values
y2 = df.loc[mask, "Solar_Cooking_Count"].values
kf2 = KFold(n_splits=5, shuffle=True, random_state=42)
rf4 = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=2, random_state=42)
lin4 = LinearRegression()
rf_pred2 = cross_val_predict(rf4, Xm2, y2, cv=kf2)
lin_pred2 = cross_val_predict(lin4, Xm2, y2, cv=kf2)
rf_metrics_nokwe = report("Random Forest (no Kweneng East)", y2, rf_pred2)
lin_metrics_nokwe = report("Linear Regression (no Kweneng East)", y2, lin_pred2)

with open("deeper_checks_results.txt", "w") as f:
    f.write("Deeper checks: naive baseline, second ensemble method, bootstrap CI, outlier-removed refit\n\n")
    f.write(f"Naive (mean rate x population):   R2={naive_metrics[0]:.3f}  RMSE={naive_metrics[1]:.2f}  MAE={naive_metrics[2]:.2f}\n")
    f.write(f"Random Forest (5-fold, count):    R2={rf_metrics[0]:.3f}  RMSE={rf_metrics[1]:.2f}  MAE={rf_metrics[2]:.2f}\n")
    f.write(f"Linear Regression (5-fold, count):R2={lin_metrics[0]:.3f}  RMSE={lin_metrics[1]:.2f}  MAE={lin_metrics[2]:.2f}\n")
    f.write(f"Gradient Boosting (5-fold, count):R2={gbr_metrics[0]:.3f}  RMSE={gbr_metrics[1]:.2f}  MAE={gbr_metrics[2]:.2f}\n\n")
    f.write(f"Bootstrap 95% CI, RMSE(RF)-RMSE(Linear): [{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}]  (RF worse in {pct_rf_worse_rmse:.1f}% of resamples)\n")
    f.write(f"Bootstrap 95% CI, R2(Linear)-R2(RF): [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]  (Linear better in {pct_linear_better_r2:.1f}% of resamples)\n")
    f.write(f"Bootstrap 95% CI, MAE(RF)-MAE(Linear): [{mae_ci[0]:.2f}, {mae_ci[1]:.2f}]\n\n")
    f.write(f"Excluding Kweneng East:\n")
    f.write(f"  Random Forest: R2={rf_metrics_nokwe[0]:.3f}  RMSE={rf_metrics_nokwe[1]:.2f}  MAE={rf_metrics_nokwe[2]:.2f}\n")
    f.write(f"  Linear Regression: R2={lin_metrics_nokwe[0]:.3f}  RMSE={lin_metrics_nokwe[1]:.2f}  MAE={lin_metrics_nokwe[2]:.2f}\n")

print("\nWrote deeper_checks_results.txt")
