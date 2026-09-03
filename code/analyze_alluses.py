import numpy as np
np.random.seed(42)
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran
from esda.getisord import G_Local

df = pd.read_csv("botswana_solar_2022.csv")
coords = df[["Longitude", "Latitude"]].values
w = KNN.from_array(coords, k=5)
w.transform = "r"

def classify(z, p):
    if p > 0.05:
        return "n.s."
    return "Hot" if z > 0 else "Cold"

results = {}
for label, col in [
    ("Cooking", "Solar_Cooking_Pct"),
    ("Heating space", "Solar_HeatingSpace_Pct"),
    ("Heating water", "Solar_HeatingWater_Pct"),
]:
    y = df[col].values
    mi = Moran(y, w, permutations=9999)
    gi = G_Local(y, w, transform="r", permutations=9999)
    cls = [classify(z, p) for z, p in zip(gi.Zs, gi.p_sim)]
    hot = df.loc[[c == "Hot" for c in cls], "District"].tolist()
    cold = df.loc[[c == "Cold" for c in cls], "District"].tolist()
    results[label] = dict(I=mi.I, EI=mi.EI, p=mi.p_sim, z=mi.z_sim, hot=hot, cold=cold)
    print(f"\n{label}: Moran's I={mi.I:.4f} (E[I]={mi.EI:.4f}), p={mi.p_sim:.4f}, z={mi.z_sim:.4f}")
    print("  Hotspots:", hot)
    print("  Coldspots:", cold)

# Consistency check: how many districts are hotspots/coldspots across >=2 of the 3 indicators?
import itertools
all_districts = df["District"].tolist()
hot_counts = {d: 0 for d in all_districts}
cold_counts = {d: 0 for d in all_districts}
for label, r in results.items():
    for d in r["hot"]:
        hot_counts[d] += 1
    for d in r["cold"]:
        cold_counts[d] += 1

print("\nDistricts that are hotspots in >=2 of 3 end-uses:", [d for d, c in hot_counts.items() if c >= 2])
print("Districts that are coldspots in >=2 of 3 end-uses:", [d for d, c in cold_counts.items() if c >= 2])

with open("alluses_summary.txt", "w") as f:
    for label, r in results.items():
        f.write(f"{label}: I={r['I']:.4f}, p={r['p']:.4f}, z={r['z']:.4f}\n")
        f.write(f"  Hot: {r['hot']}\n  Cold: {r['cold']}\n")
    f.write(f"\nConsistent hotspots (>=2/3): {[d for d, c in hot_counts.items() if c >= 2]}\n")
    f.write(f"Consistent coldspots (>=2/3): {[d for d, c in cold_counts.items() if c >= 2]}\n")
