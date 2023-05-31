import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# Path to the file where edge geometries are stored.
EDGE_FILENAME = "../MetropolisIDF/output/here_network/edges.fgb"


def get_edges():
    edges = gpd.read_file(EDGE_FILENAME)
    return edges


def get_stats(edges):
    print("Total length of the network: {} km".format(edges["length"].sum() / 1000))
    print(
        "Length of the main network: {} km".format(edges.loc[edges["main"], "length"].sum() / 1000)
    )


def plot_lanes_hist(edges, mask=None):
    if mask is None:
        mask = edges["main"]
    fig, ax = plt.subplots()
    M = edges.loc[mask, "lanes"].max()
    bins = np.arange(1, M + 1)
    counts, bins = np.histogram(edges.loc[mask, "lanes"], bins=bins, density=True)
    if M > 5:
        # Sum the lanes >= 5 as one single bin.
        bins = np.arange(1, 6)
        counts = np.concatenate((counts[:4], [np.sum(counts[4:])]))
    ax.bar(bins, counts)
    ax.set_xlabel("Number of lanes")
    ax.set_ylabel("Density")
    ax.set_xlim(0.5, min(M, 5) + 0.5)
    ax.set_ylim(0)
    ax.set_xticks(bins)
    if M > 5:
        ax.set_xticklabels(["1", "2", "3", "4", "5+"])
    ax.set_title("Number of lanes")
    fig.tight_layout()
    return fig


def plot_capacity_hist(edges, mask=None):
    if mask is None:
        mask = edges["main"]
    tot_length = edges.loc[mask, "length"].sum() / 1000
    fig, ax = plt.subplots()
    capacities = edges.loc[mask].groupby("capacity")["length"].sum() / 1000
    ax.bar(capacities.index, capacities / tot_length, width=140)
    ax.set_xlabel("Capacity (PCE / hour / lane)")
    ax.set_ylabel("Density (weighted by length)")
    ax.set_ylim(0)
    ax.set_title("Capacity")
    fig.tight_layout()
    return fig


def plot_speed_hist(edges, mask=None):
    if mask is None:
        mask = edges["main"]
    tot_length = edges.loc[mask, "length"].sum() / 1000
    fig, ax = plt.subplots()
    speeds = edges.loc[mask].groupby("speed")["length"].sum() / 1000
    ax.bar(speeds.index, speeds / tot_length, width=9)
    ax.set_xlabel("Speed (km / h)")
    ax.set_ylabel("Density (weighted by length)")
    ax.set_xlim(0, 140)
    ax.set_ylim(0)
    ax.set_title("Speed")
    fig.tight_layout()
    return fig


def plot_road_type_chart(edges, mask=None):
    if mask is None:
        mask = edges["main"]
    names = {
        1: "Motorway",
        2: "Trunk",
        3: "Primary",
        4: "Secondary",
        5: "Tertiary",
        6: "Motorway",
        7: "Trunk",
        8: "Primary",
        9: "Secondary",
        10: "Tertiary",
        11: "Living street",
        12: "Unclassified",
        13: "Residential",
    }
    edges["road_type_name"] = edges["road_type"].apply(lambda rt: names[rt])
    road_types = edges.loc[mask].groupby("road_type_name")["length"].sum() / 1000
    # Sort index.
    road_types = road_types[
        [
            "Motorway",
            "Trunk",
            "Primary",
            "Secondary",
            "Tertiary",
            "Living street",
            "Unclassified",
            "Residential",
        ]
    ]
    fig, ax = plt.subplots()
    ax.pie(road_types, labels=road_types.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Highway types (share of length)")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
