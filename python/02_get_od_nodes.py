import time

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from tqdm import tqdm

tqdm.pandas()

# Path to the CSV file with the trips of the synthetic population (including x0, y0, x1 and y1
# columns).
TRIPS_FILE = "./data/synthetic_population/ile_de_france_trips.csv"
# Returns only trips whose mode is within the following modes
# (available values: car, car_passenger, pt, walk, bike).
MODES = ("car", "car_passenger", "pt", "walk", "bike")
# Returns only trips whose departure time is later than this value (in seconds after midnight).
START_TIME = 3.0 * 3600.0
# Returns only trips whose arrival time is earlier than this value (in seconds after midnight).
END_TIME = 10.0 * 3600.0

# Path to the FlatGeobuf file where edges are stored.
EDGE_FILE = "./output/osm_network/osm_edges.fgb"
# CRS to use for metric operations.
METRIC_CRS = "EPSG:2154"

# Path to the files where the output trips should be stored.
OUTPUT_FILE = "./output/trips/trips.csv"


def prepare_trips():
    print("Reading trips")
    trips = pd.read_csv(TRIPS_FILE)
    trips = trips.loc[
        trips["mode"].isin(MODES)
        & (trips["departure_time"] >= START_TIME)
        & (trips["arrival_time"] <= END_TIME)
    ]
    trips.reset_index(inplace=True, names="trip_id")
    trips = gpd.GeoDataFrame(trips)
    print("Creating geometries")
    trips["origin"] = gpd.GeoSeries.from_xy(trips["x0"], trips["y0"], crs=METRIC_CRS)
    trips["destination"] = gpd.GeoSeries.from_xy(trips["x1"], trips["y1"], crs=METRIC_CRS)

    # separation of Origin and Destination gdf:
    origins = gpd.GeoDataFrame(trips["trip_id"], geometry=trips["origin"])
    destinations = gpd.GeoDataFrame(trips["trip_id"], geometry=trips["destination"])

    return trips, origins, destinations


def read_edges():
    print("Reading edges")
    edges = gpd.read_file(EDGE_FILE)
    edges.to_crs(METRIC_CRS, inplace=True)
    edges.sort_values('index', inplace=True)
    # Return only the edges that can be used as origin / destination edge.
    edges = edges.loc[edges["allow_od"]].copy()
    # Create source and target points.
    edges["source_point"] = edges["geometry"].apply(lambda g: Point(g.coords[0]))
    edges["target_point"] = edges["geometry"].apply(lambda g: Point(g.coords[-1]))
    return edges


def nearjoin(gdf, edges):
    assert gdf.crs == METRIC_CRS
    assert edges.crs == METRIC_CRS

    print("Join nearest edges...")
    nearedges = gdf.sjoin_nearest(
        edges[["index", "source", "source_point", "target", "target_point", "geometry"]],
        distance_col="edge_dist",
        how="inner",
    )
    nearedges.drop_duplicates(subset=["trip_id"], inplace=True)

    print("Compute distance node ...")
    # set source / target point distance
    print("  sources")
    nearedges["source_dist"] = nearedges.progress_apply(
        lambda e: e["geometry"].distance(e["source_point"]), axis=1
    )
    print("  targets")
    nearedges["target_dist"] = nearedges.progress_apply(
        lambda e: e["geometry"].distance(e["target_point"]), axis=1
    )
    # True when source closer, False when target is closer.
    print("Find nearest node...")
    nearedges["nearest"] = nearedges["source_dist"] < nearedges["target_dist"]
    # set connector value acconrdingly
    print("Assignating values:")
    nearedges["connect"] = None
    nearedges["connect_dist"] = None
    nearedges["geometry"] = None
    print("-when source is near")
    mask = nearedges["nearest"]
    nearedges.loc[mask, "connect"] = nearedges.loc[mask, "source"]
    nearedges.loc[mask, "connect_dist"] = nearedges.loc[mask, "source_dist"]
    nearedges.loc[mask, "geometry"] = nearedges.loc[mask, "source_point"]
    print("-when target is near")
    mask = ~nearedges["nearest"]
    nearedges.loc[mask, "connect"] = nearedges.loc[mask, "target"]
    nearedges.loc[mask, "connect_dist"] = nearedges.loc[mask, "target_dist"]
    nearedges.loc[mask, "geometry"] = nearedges.loc[mask, "target_point"]

    nearedges.drop(
        columns=[
            "source",
            "source_point",
            "source_dist",
            "target",
            "target_point",
            "target_dist",
            "nearest",
        ],
        inplace=True,
    )
    nearedges.set_index("trip_id", inplace=True)
    print("Done!")
    return nearedges


if __name__ == "__main__":
    t0 = time.time()

    trips, origins, destinations = prepare_trips()
    edges = read_edges()

    nearest_origins = nearjoin(origins, edges)
    nearest_destinations = nearjoin(destinations, edges)

    trips = trips.join(
        nearest_origins[["connect", "connect_dist", "edge_dist"]].add_prefix("O_"),
        how="inner",
        on="trip_id",
    )
    trips = trips.join(
        nearest_destinations[["connect", "connect_dist", "edge_dist"]].add_prefix("D_"),
        how="inner",
        on="trip_id",
    )

    # delete trips with same O and D
    n = len(trips)
    trips = trips[trips["O_connect"] != trips["D_connect"]]
    nb_removed = n - len(trips)
    if nb_removed > 0:
        print("Warning: removed {} round trips (same origin and destination)".format(nb_removed))

    trips.to_csv(OUTPUT_FILE, index=False)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
