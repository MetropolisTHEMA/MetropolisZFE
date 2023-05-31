import json
from collections import defaultdict

import geopandas as gpd
import zstandard as zstd

# Path to the output agent results file.
AGENT_RESULTS_FILENAME = "./output/runs/40/output/agent_results.json.zst"
# Path to the file where edge geometries are stored.
EDGE_FILENAME = "../MetropolisIDF/output/here_network/edges.fgb"


def get_agent_results():
    dctx = zstd.ZstdDecompressor()
    with open(AGENT_RESULTS_FILENAME, "br") as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data


def get_edges():
    edges = gpd.read_file(EDGE_FILENAME)
    return edges


def get_flow_counts(agent_results):
    flows = defaultdict(lambda: 0)
    for agent in agent_results:
        for leg in agent['mode_results']['value']['legs']:
            if leg['class']['type'] != 'Road':
                continue
            for edge in leg['class']['value']['route']:
                flows[edge[0]] += 1
    return flows
