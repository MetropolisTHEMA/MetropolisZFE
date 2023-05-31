import json

import geopandas as gpd

# Path to the file where the edge files are stored.
EDGES_FILE = "./output/osm_network/osm_edges_server.fgb"
# Path to the output file.
OUTPUT_FILE = "./output/routing/routing_graph.json"

print("Reading edges")
edges = gpd.read_file(EDGES_FILE)

edges = edges.loc[edges['main']].copy()
edges.sort_values('index_main', inplace=True)

print("Creating routing graph")
metro_edges = list()
for _, row in edges.iterrows():
    edge = [
        row["source"],
        row["target"],
        row['length'] / (row['speed'] / 3.6), # Travel time in seconds.
    ]
    metro_edges.append(edge)

print("Writing data...")
with open(OUTPUT_FILE, "w") as f:
    f.write(json.dumps(metro_edges))
