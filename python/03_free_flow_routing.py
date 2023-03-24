import os
import time
import json
import subprocess

import pandas as pd
import geopandas as gpd

# Path to the files where the trips are stored.
TRIPS_FILE = "./output/trips/trips.csv"

# Path to the FlatGeobuf file where edges are stored.
EDGE_FILE = "./output/osm_network/osm_edges.fgb"
# CRS to use for metric operations.
METRIC_CRS = "EPSG:2154"

# Directory where temporary files for the routing script should be stored.
ROUTING_DIR = "./output/routing/"
# Routing script path.
SCRIPT_FILE = "./execs/compute_travel_times"
# Parameters for the routing script.
PARAMETERS = {
    "algorithm": "Best",
    "output_route": True,
}


def read_trips():
    print("Reading trips")
    return pd.read_csv(TRIPS_FILE)


def read_edges():
    print("Reading edges")
    edges = gpd.read_file(EDGE_FILE)
    edges.to_crs(METRIC_CRS, inplace=True)
    edges.sort_values('index', inplace=True)

    # Computes edges' travel time in seconds.
    # `length` is in meters and `speed` is in km/h
    edges["tt"] = (edges["length"] / edges["speed"]) * 3.6

    # Add a penalty on non-main edges so that they are less likely to be taken in the middle of the
    # trip.
    edges.loc[~edges["main_graph"], "tt"] += 5.0

    return edges


def prepare_shortestpath(trips, edges):
    print("Creating queries")
    trips["departure_time"] = 0.0
    columns = ["trip_id", "O_connect", "D_connect", "departure_time"]
    queries = [*map(tuple, zip(*map(trips.get, columns)))]

    print("Creating Graph")
    columns = ["source", "target", "tt"]
    graph = [*map(tuple, zip(*map(edges.get, columns)))]

    print("Writing data...")
    if not os.path.isdir(ROUTING_DIR):
        os.makedirs(ROUTING_DIR)
    print("Queries")
    with open(os.path.join(ROUTING_DIR, "queries.json"), "w") as f:
        json.dump(queries, f)
    print("Graph")
    with open(os.path.join(ROUTING_DIR, "graph.json"), "w") as f:
        json.dump(graph, f)
    print("Parameters")
    with open(os.path.join(ROUTING_DIR, "parameters.json"), "w") as f:
        json.dump(PARAMETERS, f)

    print("Done!")


def run_shortestpath():

    # Inputs of the shortestpath script

    # Path to the file where the queries to compute are stored
    q = "--queries  {}".format(os.path.join(ROUTING_DIR, "queries.json"))
    # Path to the file where the graph is stored
    g = "--graph {}".format(os.path.join(ROUTING_DIR, "graph.json"))
    # Path to the file where the parameters are stored
    p = "--parameters {}".format(os.path.join(ROUTING_DIR, "parameters.json"))
    # Path to the file where the results of the queries should be stored
    o = "--output {}".format(os.path.join(ROUTING_DIR, "output.json"))

    # Run TCH script
    print("Run TCH script")

    command = " ".join((SCRIPT_FILE, q, g, p, o))

    subprocess.run(command, shell=True)
    print("Done!")


if __name__ == "__main__":

    t0 = time.time()

    trips = read_trips()

    edges = read_edges()

    prepare_shortestpath(trips, edges)

    run_shortestpath()

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
