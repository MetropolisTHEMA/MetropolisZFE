import os
import time
import json
import subprocess

import pandas as pd
import geopandas as gpd

# Path to the files where the trips are stored.
TRIPS_FILE = "./output/trips/trips.csv"
# Crit'air labels which are forbidden in the ZFE.
INVALID_CRITAIRS = ("Crit'air 4", "Crit'air 5", "Inconnu", "Non classée")

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
    "output_route": False,
}


def read_trips():
    print("Reading trips")
    trips = pd.read_csv(TRIPS_FILE)
    print("{} trips read".format(len(trips)))
    return trips


def read_edges():
    print("Reading edges")
    edges = gpd.read_file(EDGE_FILE)
    # Select main edges, not in the ZFE.
    edges = edges.loc[edges["main"] & (~edges["zfe"])].copy()
    edges.sort_values("index_main", inplace=True)
    # Computes edges' travel time in seconds.
    # `length` is in meters and `speed` is in km/h
    edges["tt"] = (edges["length"] / edges["speed"]) * 3.6
    return edges


def prepare_shortestpath(trips, edges):
    print("Creating queries")
    # Select trips with an invalid vehicle.
    polluting_trips = trips.loc[trips["critair"].isin(INVALID_CRITAIRS)].copy()
    polluting_trips = polluting_trips.loc[polluting_trips['road_leg']].copy()
    polluting_trips["departure_time"] = 0.0
    polluting_trips["O_node"] = polluting_trips["O_node"].astype(int)
    polluting_trips["D_node"] = polluting_trips["D_node"].astype(int)
    columns = ["trip_id", "O_node", "D_node", "departure_time"]
    queries = [*map(tuple, zip(*map(polluting_trips.get, columns)))]

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


def load_results():
    print("Reading routing results")
    with open(os.path.join(ROUTING_DIR, "output.json"), "r") as f:
        data = json.load(f)
    df = pd.DataFrame(list(filter(None, data["results"])), columns=["trip_id", "travel_time"])
    print("{} trips are not feasible".format(len(data['results']) - len(df)))
    return df


if __name__ == "__main__":

    t0 = time.time()

    trips = read_trips()

    edges = read_edges()

    prepare_shortestpath(trips, edges)

    run_shortestpath()

    results = load_results()

    trips['is_feasible'] = True
    trips.loc[trips['road_leg'] & trips['critair'].isin(INVALID_CRITAIRS), 'is_feasible'] = False
    trips.loc[trips['trip_id'].isin(results['trip_id']), 'is_feasible'] = True

    print("Writing trips...")
    trips.to_csv(TRIPS_FILE, index=False)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
