import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the file where the trips to simulate are stored.
TRIP_FILE = "./output/trips/trips.csv"
# Path to the file where the Commune-level vehicles types are stored.
VEHICLE_FILE = "./data/parc/Parc_VP_Communes_2021.xlsx"
# Path to the file with the Commune geometries.
COMMUNE_FILE = "./data/contours_communes/communes-20210101.shp"
# Crit'air labels which are forbidden in the ZFE.
INVALID_CRITAIRS = ("Crit'air 4", "Crit'air 5", "Inconnu", "Non classée")
# CRS to use for metric operations.
METRIC_CRS = "EPSG:2154"

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

print("Reading communes...")
communes = gpd.read_file(COMMUNE_FILE)
communes.drop(columns=["wikipedia", "surf_ha"], inplace=True)
communes.to_crs(METRIC_CRS, inplace=True)

# Save a FlatGeobuf with the communes geometries and the invalid share.
gdf = communes.merge(invalid_share, left_on="insee", right_index=True, how="right")
gdf.to_file(OUTPUT_COMMUNE, driver="FlatGeobuf")

print("Reading trips...")
trips = pd.read_csv(TRIP_FILE)
trips.sort_values(["person_id", "trip_index"], inplace=True)

print("Finding INSEE commune of origin...")
homes = trips.groupby("person_id")[["x0", "y0"]].first().reset_index()
homes = gpd.GeoDataFrame(homes, geometry=gpd.GeoSeries.from_xy(homes['x0'], homes['y0']))
homes.set_crs(METRIC_CRS, inplace=True)
homes = homes.sjoin(communes, how='left', predicate='within')
homes = homes.groupby('person_id')['insee'].first().reset_index()

homes["critair"] = None

print("Drawing vehicles...")
for insee_code, idx in homes.groupby("insee").groups.items():
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
