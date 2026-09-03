import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv("botswana_solar_2022_with_stats.csv")
pred = pd.read_csv("predictions.csv")

COLORS = {"Hot spot": "#D64550", "Cold spot": "#2E6F95", "Not significant": "#B9B9B9"}

# ---------------- Figure 1: hotspot point map ----------------
fig, ax = plt.subplots(figsize=(7.2, 7.6), dpi=200)

for cls, sub in df.groupby("Hotspot_class"):
    ax.scatter(
        sub["Longitude"], sub["Latitude"],
        s=60 + sub["Solar_Cooking_Pct"] * 220,
        c=COLORS[cls], edgecolor="white", linewidth=0.7,
        alpha=0.9, zorder=3, label=cls,
    )

for _, r in df.iterrows():
    ax.annotate(
        r["District"], (r["Longitude"], r["Latitude"]),
        fontsize=6.3, xytext=(4, 3), textcoords="offset points", color="#333333",
    )

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(
    "Getis-Ord Gi* Hotspot Classification of Solar-for-Cooking Adoption\nBotswana Census Sub-Districts, 2022 (point size = adoption %)",
    fontsize=10.5,
)
ax.set_facecolor("#F7F7F5")
ax.grid(True, linewidth=0.4, alpha=0.5)

handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[k], markersize=9, label=k) for k in ["Hot spot", "Cold spot", "Not significant"]]
ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8.5, title="Gi* class (p < .05)")

plt.tight_layout()
plt.savefig("fig1_hotspot_map.png", dpi=200)
plt.close()

# ---------------- Figure 2: actual vs predicted (RF vs Linear, CV out-of-fold) ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=200, sharex=True, sharey=True)

lims = [0, max(pred["Solar_Cooking_Count"].max(), pred["RF_Predicted"].max(), pred["Linear_Predicted"].max()) * 1.08]

for ax, col, title, color in zip(
    axes, ["RF_Predicted", "Linear_Predicted"],
    ["Random Forest", "Linear Regression"], ["#D64550", "#2E6F95"],
):
    ax.plot(lims, lims, linestyle="--", color="#888888", linewidth=1, label="Perfect prediction")
    ax.scatter(pred["Solar_Cooking_Count"], pred[col], color=color, s=45, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Actual solar-for-cooking households")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linewidth=0.4, alpha=0.5)

axes[0].set_ylabel("Predicted (5-fold cross-validated)")
fig.suptitle("Actual vs. Cross-Validated Predicted Solar-Cooking Adoption by District", fontsize=11.5)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("fig2_actual_vs_predicted.png", dpi=200)
plt.close()

print("Figures written.")
