import numpy as np
np.random.seed(42)
import pandas as pd
import statsmodels.api as sm
from libpysal.weights import KNN
from esda.moran import Moran
from esda.getisord import G_Local
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("botswana_solar_2022.csv")
n = len(df)
print("N =", n)

# ---------- 1. Descriptive statistics (fixes the thesis's broken Table 1) ----------
desc = df[["Latitude", "Longitude", "Solar_Cooking_Pct", "Total_Households", "Solar_Cooking_Count"]].describe().T
desc.to_csv("descriptive_stats.csv")
print("\n--- Descriptive statistics ---")
print(desc)

# ---------- 2. Spatial autocorrelation: Global Moran's I + local Getis-Ord Gi* ----------
coords = df[["Longitude", "Latitude"]].values
w = KNN.from_array(coords, k=5)
w.transform = "r"

y = df["Solar_Cooking_Pct"].values
mi = Moran(y, w, permutations=9999)
print("\n--- Global Moran's I (Solar_Cooking_Pct, k=5 NN weights) ---")
print(f"Moran's I = {mi.I:.4f}, expected I = {mi.EI:.4f}, p (sim) = {mi.p_sim:.4f}, z = {mi.z_sim:.4f}")

gi = G_Local(y, w, transform="r", permutations=9999)
df["Gi_zscore"] = gi.Zs
df["Gi_pvalue"] = gi.p_sim


def classify(z, p):
    if p > 0.05:
        return "Not significant"
    return "Hot spot" if z > 0 else "Cold spot"


df["Hotspot_class"] = [classify(z, p) for z, p in zip(df["Gi_zscore"], df["Gi_pvalue"])]
print("\n--- Local Getis-Ord Gi* classification ---")
print(df[["District", "Solar_Cooking_Pct", "Gi_zscore", "Gi_pvalue", "Hotspot_class"]].sort_values("Gi_zscore", ascending=False).to_string(index=False))

# ---------- 3. Regression: proportion outcome -> Binomial GLM (weights = Total_Households) ----------
X = sm.add_constant(df[["Latitude", "Longitude", "Urban"]])
success = df["Solar_Cooking_Count"]
fail = df["Total_Households"] - df["Solar_Cooking_Count"]
endog = np.column_stack([success, fail])
glm_binom = sm.GLM(endog, X, family=sm.families.Binomial())
res_binom = glm_binom.fit()
print("\n--- Binomial GLM: Solar_Cooking_Count / Total_Households ~ Latitude + Longitude + Urban ---")
print(res_binom.summary())
with open("glm_binomial_summary.txt", "w") as f:
    f.write(str(res_binom.summary()))

# Pseudo R^2 (McFadden) for comparability with the thesis's Poisson pseudo-R^2
null_X = sm.add_constant(np.ones(n))
null_model = sm.GLM(endog, X[["const"]], family=sm.families.Binomial()).fit()
pseudo_r2 = 1 - res_binom.llf / null_model.llf
print(f"\nMcFadden pseudo-R^2 = {pseudo_r2:.4f}")

# ---------- 4. Predictive modelling: Random Forest vs. linear baseline, 5-fold CV ----------
feat_cols = ["Latitude", "Longitude", "Total_Households", "Urban"]
Xm = df[feat_cols].values
ym = df["Solar_Cooking_Count"].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestRegressor(n_estimators=500, max_depth=4, min_samples_leaf=2, random_state=42)
lin = LinearRegression()

rf_pred = cross_val_predict(rf, Xm, ym, cv=kf)
lin_pred = cross_val_predict(lin, Xm, ym, cv=kf)

def report(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name}: R2={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}")
    return r2, rmse, mae

print("\n--- 5-fold cross-validated predictive performance (target: Solar_Cooking_Count) ---")
rf_metrics = report("Random Forest", ym, rf_pred)
lin_metrics = report("Linear Regression", ym, lin_pred)

rf.fit(Xm, ym)
importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\n--- Random Forest feature importances (fit on full data) ---")
print(importances)

# Save predictions for plotting
pred_df = df[["District", "Solar_Cooking_Count"]].copy()
pred_df["RF_Predicted"] = rf_pred
pred_df["Linear_Predicted"] = lin_pred
pred_df.to_csv("predictions.csv", index=False)

df.to_csv("botswana_solar_2022_with_stats.csv", index=False)

with open("results_summary.txt", "w") as f:
    f.write(f"N = {n}\n")
    f.write(f"Global Moran's I = {mi.I:.4f} (E[I]={mi.EI:.4f}), p_sim={mi.p_sim:.4f}, z_sim={mi.z_sim:.4f}\n")
    f.write(f"McFadden pseudo-R2 (binomial GLM) = {pseudo_r2:.4f}\n")
    f.write(f"RF: R2={rf_metrics[0]:.3f}, RMSE={rf_metrics[1]:.1f}, MAE={rf_metrics[2]:.1f}\n")
    f.write(f"Linear: R2={lin_metrics[0]:.3f}, RMSE={lin_metrics[1]:.1f}, MAE={lin_metrics[2]:.1f}\n")
    f.write("Feature importances:\n" + importances.to_string() + "\n")
    f.write("\nHotspot classification:\n")
    f.write(df[["District","Solar_Cooking_Pct","Gi_zscore","Gi_pvalue","Hotspot_class"]].sort_values("Gi_zscore", ascending=False).to_string(index=False))

print("\nDone.")
