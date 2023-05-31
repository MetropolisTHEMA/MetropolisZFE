import json

import geopandas as gpd
import folium
from folium.vector_layers import Circle, PolyLine
from matplotlib import colormaps
from matplotlib.colors import to_hex
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


def get_agent_map(agent, edges):
    assert agent["mode_results"]["type"] == "Trip"
    legs = list()
    for leg in agent["mode_results"]["value"]["legs"]:
        if leg["class"]["type"] != "Road":
            continue
        legs.append({"arrival_time": leg["arrival_time"], "route": leg["class"]["value"]["route"]})

    edges_taken_set = {e[0] for leg in legs for e in leg["route"]}
    edges_taken = edges.loc[list(edges_taken_set)]

    centroids = edges_taken.centroid.to_crs("epsg:4326")
    mean_location = [centroids.y.mean(), centroids.x.mean()]

    m = folium.Map(
        location=mean_location,
        zoom_start=13,
        tiles="https://api.maptiler.com/maps/basic-v2/256/{z}/{x}/{y}.png?key=ReELeWjLPpebJEd9Ss1D",
        attr='\u003ca href="https://www.maptiler.com/copyright/" target="_blank"\u003e\u0026copy; MapTiler\u003c/a\u003e \u003ca href="https://www.openstreetmap.org/copyright" target="_blank"\u003e\u0026copy; OpenStreetMap contributors\u003c/a\u003e',
    )

    edges_taken.to_crs("epsg:4326", inplace=True)
    edges_taken["fftt"] = edges_taken["length"] / (edges_taken["speed"] / 3.6)

    colormap = colormaps["RdYlGn"]

    for leg in legs:
        origin_coords = edges_taken.loc[leg["route"][0][0], "geometry"].coords[0][::-1]
        destination_coords = edges_taken.loc[leg["route"][-1][0], "geometry"].coords[-1][::-1]
        Circle(
            location=origin_coords,
            radius=30,
            tooltip="Origin",
            opacity=0.7,
            fill_opacity=0.7,
            color="#E52424",
            fill_color="#E52424",
        ).add_to(m)
        Circle(
            location=destination_coords,
            radius=30,
            tooltip="Destination",
            opacity=0.7,
            fill_opacity=0.7,
            color="#245CE5",
            fill_color="#E52424",
        ).add_to(m)

        for i in range(len(leg["route"])):
            edge = edges_taken.loc[leg["route"][i][0]]
            edge_coords = list(edge["geometry"].coords)
            edge_coords = [p[::-1] for p in edge_coords]
            if i + 1 < len(leg["route"]):
                edge_exit = leg["route"][i + 1][1]
            else:
                edge_exit = leg["arrival_time"]
            edge_tt = edge_exit - leg["route"][i][1]
            congestion = edge["fftt"] / edge_tt
            color = to_hex(colormap(congestion))
            tooltip = "From {} to {}<br>Travel time: {}<br>Free-flow tt: {}".format(
                get_time_str(leg["route"][i][1]),
                get_time_str(edge_exit),
                get_tt_str(edge_tt),
                get_tt_str(edge["fftt"]),
            )
            PolyLine(
                locations=edge_coords, tooltip=tooltip, opacity=0.5, color=color, weight=10
            ).add_to(m)
    return m


def get_time_str(seconds_after_midnight):
    t = round(seconds_after_midnight)
    hours = t // 3600
    remainder = t % 3600
    minutes = remainder // 60
    seconds = remainder % 60
    return "{:02}:{:02}:{:02}".format(hours, minutes, seconds)


def get_tt_str(seconds):
    t = round(seconds)
    minutes = int(t // 60)
    seconds = t % 60
    return "{:01}'{:02}''".format(minutes, seconds)
