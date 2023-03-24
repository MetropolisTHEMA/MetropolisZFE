from collections import defaultdict
import os
import time
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

# library to replace apply with progress_apply and prompt the progress of the pandas operation
from tqdm import tqdm

tqdm.pandas()

# Path to the FlatGeobuf file where edges are stored.
EDGE_FILE = "./output/osm_network/osm_edges.fgb"
# Directory where temporary files for the routing script are stored.
ROUTING_DIR = "./output/routing/"
# Path to the files where the trips are stored.
TRIPS_FILE = "./output/trips/trips.csv"


def read_edges():
    print("Reading edges")
    edges = gpd.read_file(EDGE_FILE)
    edges.set_index("index", inplace=True)
    edges.sort_index(inplace=True)

    # Computes edges' travel time in seconds.
    # `length` is in meters and `speed` is in km/h
    edges["tt"] = (edges["length"] / edges["speed"]) * 3.6

    # Add a penalty on non-main edges so that they are less likely to be taken in the middle of the
    # trip.
    edges.loc[~edges["main_graph"], "tt"] += 5.0

    return edges


def load_results():
    print("Reading routing results")
    with open(os.path.join(ROUTING_DIR, "output.json"), "r") as f:
        data = json.load(f)
    return pd.DataFrame(list(data["results"]), columns=["trip_id", "travel_time", "route"])


def read_trips():
    print("Reading trips")
    return pd.read_csv(TRIPS_FILE)


def find_connections(results, edges):
    print("Prepare edges data")
    main_edges = set(edges.loc[edges["main_graph"]].index)
    edge_time = edges["tt"].to_dict()
    edge_source = edges["source"].to_dict()
    edge_target = edges["target"].to_dict()

    print("Find the first / last main edge")

    def first_main_index(route):
        for i, x in enumerate(route):
            if x in main_edges:
                return i

    def last_main_index(route):
        for i, x in enumerate(route[::-1]):
            if x in main_edges:
                return len(route) - i - 1

    results["first"] = results["route"].progress_apply(first_main_index)
    results["last"] = results["route"].progress_apply(last_main_index)

    results["main_only"] = ~results["first"].isna()

    print("Find how many residential edges are used in the middle of main trips")
    results["middle_residential"] = None
    results.loc[results["main_only"], "middle_residential"] = results.loc[
        results["main_only"]
    ].progress_apply(
        lambda r: tuple(
            e for e in r["route"][int(r["first"]) : int(r["last"]) + 1] if not e in main_edges
        ),
        axis=1,
    )
    results["nb_residential"] = results["middle_residential"].apply(lambda x: len(x or []))
    residential_in_main = set(
        results.loc[results["nb_residential"] > 0, "middle_residential"].dropna().explode()
    )
    if residential_in_main:
        print(
            (
                "{} secondary edges are added to the main graph "
                "(they are used in the middle of the trip)"
            ).format(len(residential_in_main))
        )
        main_edges = main_edges.union(residential_in_main)

        print("Find the first / last main edge again")

        def first_main_index_again(route, prev_first):
            if np.isnan(prev_first):
                return first_main_index(route)
            for i, x in enumerate(route[: int(prev_first)][::-1]):
                if not x in main_edges:
                    return prev_first - i
            else:
                # All are main.
                return 0

        def last_main_index_again(route, prev_last):
            if np.isnan(prev_last):
                return last_main_index(route)
            for i, x in enumerate(route[int(prev_last) + 1 :]):
                if not x in main_edges:
                    return prev_last + i
            else:
                # All are main.
                return len(route) - 1

        results["first"] = results.progress_apply(
            lambda r: first_main_index_again(r["route"], r["first"]), axis=1
        )
        results["last"] = results.progress_apply(
            lambda r: last_main_index_again(r["route"], r["last"]), axis=1
        )

    print("Counting occurences of each edge")
    counts = defaultdict(lambda: 0)
    for route in results['route']:
        for e in route:
            counts[e] += 1
    counts = pd.Series(counts)
    counts.name = 'count'

    print("Put trips that never use the main network on the side")
    no_main_trips = results.loc[results["first"].isna()]
    print("There are {} trips not using the main network".format(len(no_main_trips)))
    results = results.loc[~results["first"].isna()].copy()
    assert results["last"].isna().sum() == 0
    results["first"] = results["first"].astype(int)
    results["last"] = results["last"].astype(int)

    if residential_in_main:
        print("Finding main edges remaining in the access / egress part")
        results["nb_main_in_access"] = results.progress_apply(
            lambda r: sum(1 for e in r["route"][: r["first"]] if e in residential_in_main), axis=1
        )
        results["nb_main_in_egress"] = results.progress_apply(
            lambda r: sum(1 for e in r["route"][r["last"] + 1 :] if e in residential_in_main),
            axis=1,
        )
        nb_trips = np.sum((results["nb_main_in_access"] > 0) | (results["nb_main_in_egress"] > 0))
        nb_in_access = results["nb_main_in_access"].sum()
        nb_in_egress = results["nb_main_in_egress"].sum()
        if nb_trips:
            print("There are {} trips with main edges in the access / egress part".format(nb_trips))
            print(
                "representing {} edges in the access parts and {} in the egress parts".format(
                    nb_in_access, nb_in_egress
                )
            )

    print("Find the access / egress edges")
    results["O_edge"] = results.progress_apply(lambda r: r["route"][r["first"]], axis=1)
    results["D_edge"] = results.progress_apply(lambda r: r["route"][r["last"]], axis=1)

    print("Find the access / egress nodes")
    results["O_node"] = results["O_edge"].progress_apply(lambda e: edge_source[e])
    results["D_node"] = results["D_edge"].progress_apply(lambda e: edge_target[e])

    print("Number of unique origins: {}".format(results["O_node"].nunique()))
    print("Number of unique destinations: {}".format(results["D_node"].nunique()))

    print("Compute access / egress times")
    results["access_time"] = results.progress_apply(
        lambda r: sum(edge_time[e] for e in r["route"][: r["first"]]), axis=1
    )
    results["egress_time"] = results.progress_apply(
        lambda r: sum(edge_time[e] for e in r["route"][r["last"] + 1 :]), axis=1
    )

    return main_edges, results, no_main_trips, counts


def process_edges(edges, main_edges, counts):
    # Flag main edges (used in Metropolis).
    edges["main"] = edges.index.isin(main_edges)

    # Merge counts.
    edges = edges.merge(counts, left_index=True, right_index=True, how='left')
    edges['count'] = edges['count'].fillna(0)

    # Create Metropolis indices (from 0 to n-1 for main edges).
    edges.sort_values(['main', 'index'], ascending=[False, True], inplace=True)
    edges.reset_index(inplace=True)
    edges['index_main'] = edges.index
    edges.set_index('index', inplace=True, drop=True)
    edges.sort_index(inplace=True)
    return edges


def merge(trips, main_trips, no_main_trips):
    print("Merging trips data")
    main_trips["road_leg"] = True
    no_main_trips["road_leg"] = False

    main_trips = main_trips[
        ["trip_id", "travel_time", "road_leg", "O_node", "D_node", "access_time", "egress_time"]
    ].copy()
    no_main_trips = no_main_trips[["trip_id", "travel_time", "road_leg"]].copy()

    trips_connect = pd.concat((main_trips, no_main_trips))

    trips = trips.merge(trips_connect, on="trip_id", how="left")

    trips["origin_delay"] = trips["O_connect_dist"] / (30 / 3.6) + trips["access_time"]
    trips["destination_delay"] = trips["D_connect_dist"] / (30 / 3.6) + trips["egress_time"]

    return trips


def is_strongly_connected(edges):
    # If the graph is strongly connected, all nodes need to be both source and target for at least
    # one edge.
    nb_sources = edges["source"].nunique()
    nb_targets = edges["target"].nunique()
    if nb_sources != nb_targets:
        return False
    n = len(set(edges["source"]).union(set(edges["target"])))
    if n != nb_sources:
        return False
    G = nx.DiGraph()
    G.add_edges_from(
        map(
            lambda keyrow: (keyrow[1]["source"], keyrow[1]["target"]),
            edges[["source", "target"]].iterrows(),
        )
    )
    # Find the nodes of the largest strongly connected component.
    connected_nodes = max(nx.strongly_connected_components(G), key=len)
    if len(connected_nodes) < n:
        return False
    return True


if __name__ == "__main__":

    t0 = time.time()

    edges = read_edges()

    results = load_results()

    main_edges, main_trips, no_main_trips, counts = find_connections(results, edges)

    edges = process_edges(edges, main_edges, counts)

    trips = read_trips()

    print("Checking if the main graph is strongly connected")
    if not is_strongly_connected(edges):
        print("Error: The graph is not strongly connected")

    print("Saving the edges")
    edges.to_file(EDGE_FILE, driver='FlatGeobuf')

    trips = merge(trips, main_trips, no_main_trips)

    print("Saving the trips")
    trips.to_csv(TRIPS_FILE, index=False)

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
