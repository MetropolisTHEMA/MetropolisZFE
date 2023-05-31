import json

import numpy as np
import pandas as pd
import geopandas as gpd
import zstandard as zstd
import matplotlib.pyplot as plt

# Path to the output weights results file.
WEIGHT_RESULTS_FILENAME = "./output/runs/3/weight_results.json.zst"
# Path to the file where edge geometries are stored.
EDGE_FILENAME = "./output/osm_network/osm_edges_server.fgb"
# Recording period of the simulation.
PERIOD = [3.0 * 3600.0, 10.0 * 3600.0]
# Recording interval of the simulation.
INTERVAL = 300.0


def get_weight_results():
    dctx = zstd.ZstdDecompressor()
    with open(WEIGHT_RESULTS_FILENAME, "br") as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data


def get_edges():
    edges = gpd.read_file(EDGE_FILENAME)
    return edges


def get_edge_with_congestion(edges, weights, each=INTERVAL):
    xs = np.arange(PERIOD[0], PERIOD[1] + INTERVAL, INTERVAL)
    bins = np.arange(PERIOD[0], PERIOD[1] + each, each)
    array = np.empty((len(weights["road_network"][0]), len(bins)))
    for i, w in enumerate(weights["road_network"][0]):
        if isinstance(w, float):
            array[i, :] = w
        else:
            ys = np.interp(bins, xs, np.array(w["points"]))
            assert len(ys) == len(
                bins
            ), "Weights are incompatible with the recording period and interval"
            array[i, :] = ys
    ff_weights = np.min(array, axis=1)
    array /= np.atleast_2d(ff_weights).T
    wdf = pd.DataFrame(array, columns=[f"TD{i}" for i in range(len(bins))], dtype=float)
    return pd.concat((edges, wdf), axis=1)


def plot_weight(w, bottom=None):
    xs = np.arange(PERIOD[0], PERIOD[1] + INTERVAL, INTERVAL)
    if isinstance(w, float):
        ys = np.repeat(w, len(xs))
    else:
        ys = np.round(np.array(w['points']))
        assert len(ys) == len(xs), "Weights are incompatible with the recording period and interval"
    fig, ax = plt.subplots()
    ax.plot(xs / 3600, ys, '-o')
    ax.set_xlabel('Departure time (h)')
    ax.set_ylabel('Travel time (s)')
    if not bottom is None:
        ax.set_ylim(bottom=bottom)
    fig.tight_layout()
    return fig
