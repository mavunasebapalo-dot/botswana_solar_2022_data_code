import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pred = pd.read_csv("predictions.csv")
stats = pd.read_csv("botswana_solar_2022_with_stats.csv")
df = pred.merge(stats[["District", "Total_Households"]], on="District")

# ---------- Figure: Random Forest feature importances (bar chart) ----------
importances = pd.Series(
    {"Total_Households": 0.922, "Longitude": 0.058, "Latitude": 0.018, "Urban": 0.001}
).sort_values()

fig, ax = plt.subplots(figsize=(6.2, 3.2))
bars = ax.barh(importances.index, importances.values, color="#4C72B0")
ax.set_xlabel("Mean decrease in impurity")
ax.set_title("Random Forest Feature Importances (N = 28)")
ax.set_xlim(0, 1.0)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig3_feature_importance.png", dpi=150)
plt.close()
print("Wrote fig3_feature_importance.png")

# ---------- Figure: prediction error vs district size (extrapolation diagnostic) ----------
df["RF_Error"] = df["RF_Predicted"] - df["Solar_Cooking_Count"]
df["Linear_Error"] = df["Linear_Predicted"] - df["Solar_Cooking_Count"]

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.scatter(df["Total_Households"], df["RF_Error"], color="#C44E52", label="Random Forest", s=45, alpha=0.85)
ax.scatter(df["Total_Households"], df["Linear_Error"], color="#4C72B0", label="Linear Regression", s=45, alpha=0.85, marker="^")
ax.axhline(0, color="grey", linestyle="--", linewidth=1)

# annotate the Kweneng East outlier (placed above-left to avoid the x-axis label)
kwe = df[df["District"] == "Kweneng East"].iloc[0]
ax.annotate(
    "Kweneng East",
    xy=(kwe["Total_Households"], kwe["RF_Error"]),
    xytext=(kwe["Total_Households"] * 0.62, kwe["RF_Error"] + 45),
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    fontsize=9,
    ha="center",
)

ax.set_xlabel("Total households in district")
ax.set_ylabel("Prediction error\n(Predicted − Actual solar-cooking households)")
ax.set_title("Out-of-Fold Prediction Error vs. District Size")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("fig4_error_vs_size.png", dpi=150)
plt.close()
print("Wrote fig4_error_vs_size.png")
