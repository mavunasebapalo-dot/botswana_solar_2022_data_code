import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("botswana_solar_2022.csv")
feat_cols = ["Latitude", "Longitude", "Total_Households", "Urban"]
Xm = df[feat_cols].values

def report(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name}: R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    return r2, rmse, mae

results = {}

# ---------- Check 1: Leave-one-out CV, target = Solar_Cooking_Count (raw count) ----------
print("=" * 70)
print("CHECK 1: Leave-one-out CV, target = Solar_Cooking_Count")
print("=" * 70)
y_count = df["Solar_Cooking_Count"].values
loo = LeaveOneOut()

rf = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=2, random_state=42)
lin = LinearRegression()

rf_pred_loo = cross_val_predict(rf, Xm, y_count, cv=loo)
lin_pred_loo = cross_val_predict(lin, Xm, y_count, cv=loo)

results["loo_count_rf"] = report("RF (LOO, count)", y_count, rf_pred_loo)
results["loo_count_lin"] = report("Linear (LOO, count)", y_count, lin_pred_loo)

# ---------- Check 2: 5-fold and LOO CV, target = Solar_Cooking_Pct (adoption rate) ----------
print()
print("=" * 70)
print("CHECK 2: 5-fold and LOO CV, target = Solar_Cooking_Pct (rate)")
print("=" * 70)
y_rate = df["Solar_Cooking_Pct"].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf2 = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=2, random_state=42)
lin2 = LinearRegression()
rf_pred_5f_rate = cross_val_predict(rf2, Xm, y_rate, cv=kf)
lin_pred_5f_rate = cross_val_predict(lin2, Xm, y_rate, cv=kf)
results["5f_rate_rf"] = report("RF (5-fold, rate)", y_rate, rf_pred_5f_rate)
results["5f_rate_lin"] = report("Linear (5-fold, rate)", y_rate, lin_pred_5f_rate)

rf3 = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=2, random_state=42)
lin3 = LinearRegression()
rf_pred_loo_rate = cross_val_predict(rf3, Xm, y_rate, cv=loo)
lin_pred_loo_rate = cross_val_predict(lin3, Xm, y_rate, cv=loo)
results["loo_rate_rf"] = report("RF (LOO, rate)", y_rate, rf_pred_loo_rate)
results["loo_rate_lin"] = report("Linear (LOO, rate)", y_rate, lin_pred_loo_rate)

# ---------- Check 3: nested-CV hyperparameter search for RF (count target, 5-fold outer) ----------
print()
print("=" * 70)
print("CHECK 3: Nested CV hyperparameter search for RF, target = count")
print("=" * 70)
param_grid = {
    "n_estimators": [200],
    "max_depth": [3, 4, 6],
    "min_samples_leaf": [1, 2, 4],
}
outer_kf = KFold(n_splits=5, shuffle=True, random_state=42)
tuned_preds = np.zeros_like(y_count, dtype=float)
best_params_per_fold = []
for i, (train_idx, test_idx) in enumerate(outer_kf.split(Xm)):
    print(f"  outer fold {i+1}/5 ...", flush=True)
    inner = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    inner.fit(Xm[train_idx], y_count[train_idx])
    tuned_preds[test_idx] = inner.predict(Xm[test_idx])
    best_params_per_fold.append(inner.best_params_)

results["tuned_rf_count"] = report("RF tuned (nested CV, count)", y_count, tuned_preds)
print("Best params per outer fold:", best_params_per_fold)

# ---------- Summary table ----------
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
for k, (r2, rmse, mae) in results.items():
    print(f"{k:22s}  R2={r2:6.3f}  RMSE={rmse:7.2f}  MAE={mae:7.2f}")

with open("robustness_results.txt", "w") as f:
    f.write("Robustness checks: leave-one-out CV, adoption-rate target, nested-CV RF tuning\n\n")
    for k, (r2, rmse, mae) in results.items():
        f.write(f"{k:22s}  R2={r2:6.3f}  RMSE={rmse:7.2f}  MAE={mae:7.2f}\n")
    f.write("\nBest RF params per outer fold (nested CV, count target):\n")
    for i, p in enumerate(best_params_per_fold):
        f.write(f"  fold {i+1}: {p}\n")

print("\nWrote robustness_results.txt")
