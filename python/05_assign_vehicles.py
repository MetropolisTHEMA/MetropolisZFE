import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the file where the trips to simulate are stored.
TRIP_FILE = "./output/trips/trips_filtered.csv"
# Path to the file where the Commune-level vehicles types are stored.
VEHICLE_FILE = "./data/parc/Parc_VP_Communes_2021.xlsx"
# Path to the file with the Commune geometries.
COMMUNE_FILE = "./data/contours_communes/communes-20210101.shp"
# Path to the file with the IRIS data.
IRIS_FILE = "./data/contours_iris_france/CONTOURS-IRIS.shp"
# Crit'air labels which are forbidden in the ZFE.
INVALID_CRITAIRS = ("Crit'air 4", "Crit'air 5", "Inconnu", "Non classée")

# Path to the file where the output communes FlatGeobuf with invalid shares should be stored.
OUTPUT_COMMUNE = "./output/communes_invalid_share.fgb"

print("Reading vehicles...")
vehicles = pd.read_excel(VEHICLE_FILE)
vehicles = vehicles.dropna()
# Select Île-de-France région.
vehicles = vehicles.loc[vehicles["Code région"] == "11"].copy()
# Drop "Inconnu".
vehicles = vehicles.loc[vehicles["Vignette Crit'air"] != "Inconnu"].copy()

vehicles["invalid"] = vehicles["Vignette Crit'air"].isin(INVALID_CRITAIRS)
vehicles["invalid_count"] = vehicles["invalid"] * vehicles["Parc au 01/01/2021"]

print("Creating vehicle map...")
# Compute the share of invalid vehicles by commune.
invalid_share = (
    vehicles.groupby("Code commune")["invalid_count"].sum()
    / vehicles.groupby("Code commune")["Parc au 01/01/2021"].sum()
)
invalid_share.name = "invalid_share"
# Save a FlatGeobuf with the communes geometries and the invalid share.
communes = gpd.read_file(COMMUNE_FILE)
communes.drop(columns=["wikipedia", "surf_ha"], inplace=True)
gdf = communes.merge(invalid_share, left_on="insee", right_index=True, how="right")
gdf.to_file(OUTPUT_COMMUNE, driver="FlatGeobuf")

print("Reading IRIS...")
iris = gpd.read_file(IRIS_FILE)
# Manage Paris.
iris.loc[iris["INSEE_COM"].str.startswith("75"), "INSEE_COM"] = "75056"
iris = iris.loc[iris["INSEE_COM"].isin(vehicles["Code commune"])].copy()
iris = iris[["INSEE_COM", "CODE_IRIS"]].copy()
iris["CODE_IRIS"] = iris["CODE_IRIS"].astype(int)

print("Reading trips...")
trips = pd.read_csv(TRIP_FILE)
trips = trips.merge(iris, left_on="iris_origin", right_on="CODE_IRIS").drop(columns="CODE_IRIS")
trips = trips.merge(
    iris, left_on="iris_destination", right_on="CODE_IRIS", suffixes=("_origin", "_destination")
).drop(columns="CODE_IRIS")
trips.sort_values(["person_id", "trip_index"], inplace=True)

homes = trips.groupby("person_id")["INSEE_COM_origin"].first().reset_index()
homes["critair"] = None

print("Drawing vehicles...")
for insee_code, idx in homes.groupby("INSEE_COM_origin").groups.items():
    persons = homes.loc[idx]
    vehicle_pool = vehicles.loc[
        vehicles["Code commune"] == insee_code, ["Vignette Crit'air", "Parc au 01/01/2021"]
    ]
    probs = vehicle_pool["Parc au 01/01/2021"] / vehicle_pool["Parc au 01/01/2021"].sum()
    draws = np.random.choice(vehicle_pool["Vignette Crit'air"], size=len(persons), p=probs)
    homes.loc[idx, "critair"] = draws

print("Writing trips...")
trips = trips.merge(homes[['person_id', 'critair']], on='person_id', how='left')
trips.to_csv(TRIP_FILE, index=False)
