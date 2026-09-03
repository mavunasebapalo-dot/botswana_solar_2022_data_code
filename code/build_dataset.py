import pandas as pd

# Source: Statistics Botswana, "Utilisation of Green Energy Among Households in
# Botswana" (2022 Population and Housing Census), Table 3 (cooking) and
# Appendix 1 (heating space / heating water). https://statsbots.org.bw
# Coordinates: administrative-centre / largest-settlement point for each of the
# 28 census sub-districts, compiled from Wikipedia district/settlement articles
# and standard gazetteers. These are point references for the named town, not
# polygon centroids of the (often very large) sub-district boundary -- flagged
# as a limitation in the manuscript.

rows = [
    # District,                Lat,      Lon,      Cook%, HeatSpace%, HeatWater%, TotalHH, Urban
    ("Gaborone",               -24.6581,  25.9088,  0.2,  0.3, 1.0,  82421, 1),
    ("Francistown",            -21.1670,  27.5083,  0.2,  0.3, 0.5,  33811, 1),
    ("Lobatse",                -25.2222,  25.6786,  0.1,  0.0, 0.1,   9839, 1),
    ("Selibe Phikwe",          -21.9758,  27.8496,  0.2,  0.0, 0.6,  13330, 1),
    ("Orapa",                  -21.3000,  25.3667,  0.1,  3.3, 5.7,   3049, 1),
    ("Jwaneng",                -24.6000,  24.7167,  0.1,  0.4, 1.6,   6586, 1),
    ("Sowa",                   -20.5333,  26.2167,  0.1,  0.0, 0.2,   1106, 1),
    ("Southern",               -24.9667,  25.3500,  0.3,  0.2, 0.3,  37806, 0),
    ("Barolong",               -25.8833,  25.3333,  0.1,  0.1, 0.1,  16498, 0),
    ("Ngwaketse West",         -25.0500,  23.1333,  0.3,  0.2, 0.7,   6588, 0),
    ("South East",             -24.8712,  25.8171,  0.2,  0.2, 0.8,  36327, 0),
    ("Kweneng East",           -24.4067,  25.4950,  0.3,  0.3, 0.4, 100751, 0),
    ("Kweneng West",           -24.0333,  24.9833,  0.2,  0.2, 0.3,  15920, 0),
    ("Kgatleng",               -24.4256,  26.1483,  0.2,  0.3, 0.5,  36538, 0),
    ("Central Serowe-Palapye", -22.3875,  26.7107,  0.2,  0.1, 0.3,  56992, 0),
    ("Central Mahalapye",      -23.1041,  26.8142,  0.2,  0.3, 0.4,  36683, 0),
    ("Central Bobonong",       -21.9667,  28.4333,  0.2,  0.1, 0.7,  22212, 0),
    ("Central Boteti",         -21.4167,  25.6167,  0.6,  0.2, 0.6,  21259, 0),
    ("Central Tutume",         -20.4667,  27.1333,  0.2,  0.4, 0.5,  46626, 0),
    ("North East",             -20.6167,  27.4667,  0.2,  0.1, 0.4,  20912, 0),
    ("Ngamiland East",         -19.9833,  23.4167,  0.3,  0.3, 0.5,  31591, 0),
    ("Ngamiland West",         -20.4667,  22.7167,  0.3,  0.4, 1.3,  17921, 0),
    ("Chobe",                  -17.8167,  25.1500,  0.3,  0.9, 1.6,  10124, 0),
    ("Delta",                  -18.8000,  22.4167,  1.6,  1.1, 1.1,    192, 0),
    ("Ghanzi",                 -21.5667,  21.7667,  0.5,  0.6, 1.6,  15158, 0),
    ("CKGR",                   -22.1833,  23.4667,  1.2,  1.2, 3.6,     84, 0),
    ("Kgalagadi South",        -26.0500,  22.4000,  0.4,  0.3, 0.3,   9749, 0),
    ("Kgalagadi North",        -24.0167,  21.7667,  0.2,  0.3, 0.5,   7172, 0),
]

df = pd.DataFrame(rows, columns=[
    "District", "Latitude", "Longitude", "Solar_Cooking_Pct",
    "Solar_HeatingSpace_Pct", "Solar_HeatingWater_Pct", "Total_Households", "Urban",
])

for col, pct in [
    ("Solar_Cooking_Count", "Solar_Cooking_Pct"),
    ("Solar_HeatingSpace_Count", "Solar_HeatingSpace_Pct"),
    ("Solar_HeatingWater_Count", "Solar_HeatingWater_Pct"),
]:
    df[col] = (df[pct] / 100.0 * df["Total_Households"]).round().astype(int)

df.to_csv("botswana_solar_2022.csv", index=False)
print(df.to_string(index=False))
print("\nN =", len(df))
