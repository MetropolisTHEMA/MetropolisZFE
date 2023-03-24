import sys
import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmium
from osmium.geom import WKBFactory
from geojson import Point, LineString, Feature, FeatureCollection
from haversine import haversine_vector, Unit
import pyproj
from shapely.ops import transform
from shapely.prepared import PreparedGeometry, prep

# Path to the OSM PBF file.
OSM_FILE = "./data/osm/ile-de-france-2023-03-13.osm.pbf"
# Path to the FlatGeobuf file where edges should be stored.
EDGE_FILE = "./output/osm_network/osm_edges.fgb"
# CRS to use for metric operations.
METRIC_CRS = "EPSG:2154"
# List of highway tags to consider.
# See https://wiki.openstreetmap.org/wiki/Key:highway
VALID_HIGHWAYS = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "living_street",
    "unclassified",
    "residential",
    #  "road",
    #  "service",
)
# List of highway tags to keep in the final graph.
MAIN_HIGHWAYS = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    #  "living_street",
    #  "unclassified",
    #  "residential",
    #  "road",
    #  "service",
)
# List of highway tags that cannot be used as origin / destination.
OD_FORBIDDEN = (
    "motorway",
    "trunk",
    "motorway_link",
    "trunk_link",
)
# Road type id to use for each highway tag.
ROADTYPE_TO_ID = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "motorway_link": 6,
    "trunk_link": 7,
    "primary_link": 8,
    "secondary_link": 9,
    "tertiary_link": 10,
    "living_street": 11,
    "unclassified": 12,
    "residential": 13,
    "road": 14,
    "service": 15,
}
# Default number of lanes when unspecified.
DEFAULT_LANES = {
    "motorway": 2,
    "trunk": 2,
    "primary": 1,
    "secondary": 1,
    "tertiary": 1,
    "unclassified": 1,
    "residential": 1,
    "motorway_link": 1,
    "trunk_link": 1,
    "primary_link": 1,
    "secondary_link": 1,
    "tertiary_link": 1,
    "living_street": 1,
    "road": 1,
    "service": 1,
}
# Default speed, in km/h, in rural areas.
DEFAULT_SPEED_RURAL = {
    "motorway": 130,
    "trunk": 110,
    "primary": 80,
    "secondary": 80,
    "tertiary": 80,
    "unclassified": 20,
    "residential": 30,
    "motorway_link": 90,
    "trunk_link": 70,
    "primary_link": 50,
    "secondary_link": 50,
    "tertiary_link": 50,
    "living_street": 20,
    "road": 20,
    "service": 20,
}
# Default speed, in km/h, in urban areas.
DEFAULT_SPEED_URBAN = {
    "motorway": 130,
    "trunk": 110,
    "primary": 50,
    "secondary": 50,
    "tertiary": 50,
    "unclassified": 20,
    "residential": 30,
    "motorway_link": 90,
    "trunk_link": 70,
    "primary_link": 50,
    "secondary_link": 50,
    "tertiary_link": 50,
    "living_street": 20,
    "road": 20,
    "service": 20,
}
# Capacity of the different highway types (in PCE / hour).
CAPACITY = {
    "motorway": 2000,
    "trunk": 2000,
    "primary": 1500,
    "secondary": 800,
    "motorway_link": 1500,
    "trunk_link": 1500,
    "primary_link": 1500,
    "secondary_link": 800,
    "tertiary": 600,
    "tertiary_link": 600,
    "living_street": 300,
    "unclassified": 600,
    "residential": 600,
    "road": 300,
    "service": 300,
}
# Landuse tags used to define urban areas.
# See https://wiki.openstreetmap.org/wiki/Key:landuse
URBAN_LANDUSE = (
    # A commercial zone, predominantly offices or services.
    "commercial",
    # An area being built on.
    "construction",
    # An area predominately used for educational purposes/facilities.
    "education",
    # An area with predominantly workshops, factories or warehouses.
    "industrial",
    # An area with predominantly houses or apartment buildings.
    "residential",
    # An area that encloses predominantly shops.
    "retail",
    # A smaller area of grass, usually mown and managed.
    #  "grass",
    # A place where people, or sometimes animals are buried that isn't part of a place of worship.
    #  "cemetery",
    # An area of land artificially graded to hold water.
    #  "basin",
    # Allotment gardens with multiple land parcels assigned to individuals or families for
    # gardening.
    #  "allotments",
    # A village green is a distinctive area of grassy public land in a village centre.
    "village_green",
    # An area designated for flowers.
    #  "flowerbed",
    # An open green space for general recreation, which often includes formal or informal pitches,
    # nets and so on.
    "recreation_ground",
    # Area used for military purposes.
    "military",
    # Denotes areas occupied by multiple private garage buildings.
    "garages",
    # An area used for religious purposes.
    "religious",
)


def valid_way(way):
    """Returns True if the way is a valid way to consider."""
    has_access = not "access" in way.tags or way.tags["access"] == "yes"
    return has_access and len(way.nodes) > 1 and way.tags.get("highway") in VALID_HIGHWAYS


def is_urban_area(area):
    """Returns True if the area is an urban area."""
    return area.tags.get("landuse") in URBAN_LANDUSE and (area.num_rings()[0] > 0)


class UrbanAreasReader(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.wkb_factory = WKBFactory()
        self.areas_wkb = list()

    def area(self, area):
        if not is_urban_area(area):
            return
        self.handle_area(area)

    def handle_area(self, area):
        self.areas_wkb.append(self.wkb_factory.create_multipolygon(area))

    def get_urban_area(self):
        polygons = gpd.GeoSeries.from_wkb(self.areas_wkb)
        return polygons.unary_union


class NodeReader(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.all_nodes = set()
        self.nodes: dict[int, Feature] = dict()
        self.counter = 0

    def way(self, way):
        if not valid_way(way):
            return
        self.handle_way(way)

    def handle_way(self, way):
        # Always add source and origin node.
        self.add_node(way.nodes[0])
        self.add_node(way.nodes[-1])
        self.all_nodes.add(way.nodes[0])
        self.all_nodes.add(way.nodes[-1])
        # Add the other nodes if they were already explored, i.e., they
        # intersect with another road.
        for i in range(1, len(way.nodes) - 1):
            node = way.nodes[i]
            if node in self.all_nodes:
                self.add_node(node)
            self.all_nodes.add(node)

    def add_node(self, node):
        if node.ref in self.nodes:
            # Node was already added.
            return
        if node.location.valid():
            self.nodes[node.ref] = Feature(
                geometry=Point((node.lon, node.lat)),
                properties={"id": self.counter, "osm_id": node.ref},
            )
            self.counter += 1


class EdgeReader(osmium.SimpleHandler):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = nodes
        self.edges = list()
        self.counter = 0

    def way(self, way):
        self.add_way(way)

    def add_way(self, way):

        if not valid_way(way):
            return

        road_type = way.tags.get("highway", None)
        road_type_id = ROADTYPE_TO_ID[road_type]

        name = (
            way.tags.get("name", "") or way.tags.get("addr:street", "") or way.tags.get("ref", "")
        )
        name = way.tags.get("ref", "")
        if len(name) > 50:
            name = name[:47] + "..."

        oneway = (
            way.tags.get("oneway", "no") == "yes" or way.tags.get("junction", "") == "roundabout"
        )

        # Find maximum speed if available.
        maxspeed = way.tags.get("maxspeed", "")
        speed = None
        back_speed = None
        if maxspeed == "FR:walk":
            speed = 20
        elif maxspeed == "FR:urban":
            speed = 50
        elif maxspeed == "FR:rural":
            speed = 80
        else:
            try:
                speed = float(maxspeed)
            except ValueError:
                pass
        if not oneway:
            try:
                speed = float(way.tags.get("maxspeed:forward", "0")) or speed
            except ValueError:
                pass
            try:
                back_speed = float(way.tags.get("maxspeed:backward", "0")) or speed
            except ValueError:
                pass

        # Find number of lanes if available.
        lanes = None
        back_lanes = None
        if oneway:
            try:
                lanes = int(way.tags.get("lanes", ""))
            except ValueError:
                pass
            else:
                lanes = max(lanes, 1)
        else:
            try:
                lanes = (
                    int(way.tags.get("lanes:forward", "0")) or int(way.tags.get("lanes", "")) // 2
                )
            except ValueError:
                pass
            else:
                lanes = max(lanes, 1)
            try:
                back_lanes = (
                    int(way.tags.get("lanes:backward", "0")) or int(way.tags.get("lanes", "")) // 2
                )
            except ValueError:
                pass
            else:
                back_lanes = max(back_lanes, 1)
        if lanes is None:
            lanes = DEFAULT_LANES.get(road_type, 1)
        if back_lanes is None:
            back_lanes = DEFAULT_LANES.get(road_type, 1)

        capacity = CAPACITY.get(road_type)

        for i, node in enumerate(way.nodes):
            if node.ref in self.nodes:
                source = i
                break
        else:
            # No node of the way is in the nodes.
            return

        j = source + 1
        for i, node in enumerate(list(way.nodes)[j:]):
            if node.ref in self.nodes:
                target = j + i
                self.add_edge(
                    way,
                    source,
                    target,
                    oneway,
                    name,
                    road_type_id,
                    lanes,
                    back_lanes,
                    speed,
                    back_speed,
                    capacity,
                )
                source = target

    def add_edge(
        self,
        way,
        source,
        target,
        oneway,
        name,
        road_type,
        lanes,
        back_lanes,
        speed,
        back_speed,
        capacity,
    ):
        source_id = self.nodes[way.nodes[source].ref].properties["id"]
        target_id = self.nodes[way.nodes[target].ref].properties["id"]
        if source_id == target_id:
            # Self-loop.
            return

        # Create a geometry of the road.
        coords = list()
        for i in range(source, target + 1):
            if way.nodes[i].location.valid():
                coords.append((way.nodes[i].lon, way.nodes[i].lat))
        geometry = LineString(coords)
        back_geometry = None
        if not oneway:
            back_geometry = LineString(coords[::-1])

        edge_id = self.counter
        self.counter += 1
        back_edge_id = None
        if not oneway:
            back_edge_id = self.counter
            self.counter += 1

        # Compute length in meters.
        length = np.sum(haversine_vector(coords[:-1], coords[1:], Unit.KILOMETERS)) * 1000

        self.edges.append(
            Feature(
                geometry=geometry,
                properties={
                    "id": edge_id,
                    "name": name,
                    "road_type": road_type,
                    "lanes": lanes,
                    "length": length,
                    "speed": speed,
                    "capacity": capacity,
                    "source": source_id,
                    "target": target_id,
                    "osm_id": way.id,
                },
            )
        )

        if not oneway:
            self.edges.append(
                Feature(
                    geometry=back_geometry,
                    properties={
                        "id": back_edge_id,
                        "name": name,
                        "road_type": road_type,
                        "lanes": back_lanes,
                        "length": length,
                        "speed": back_speed,
                        "capacity": capacity,
                        "source": target_id,
                        "target": source_id,
                        "osm_id": way.id,
                    },
                )
            )

    def post_process(self, urban_area: PreparedGeometry):
        edge_collection = FeatureCollection(self.edges)
        edges = gpd.GeoDataFrame.from_features(edge_collection, crs="epsg:4326")

        print("Finding the largest strongly connected component")

        G = nx.DiGraph()
        G.add_edges_from(
            map(
                lambda keyrow: (keyrow[1]["source"], keyrow[1]["target"]),
                edges[["source", "target"]].iterrows(),
            )
        )
        # Find the nodes of the largest strongly connected component.
        connected_nodes = max(nx.strongly_connected_components(G), key=len)
        if len(connected_nodes) < G.number_of_nodes():
            print(
                "Warning: discarding {} nodes disconnected from the main graph".format(
                    G.number_of_nodes() - len(connected_nodes)
                )
            )
            edges = edges.loc[
                (edges["source"].isin(connected_nodes)) & (edges["target"].isin(connected_nodes))
            ].copy()

        print("Finding parallel edges")

        # Flag the highways that we want to keep in the final graph.
        main_highways = list(map(lambda h: ROADTYPE_TO_ID[h], MAIN_HIGHWAYS))
        edges["main_graph"] = edges["road_type"].isin(main_highways)

        # Flag the highways that cannot be used as OD.
        od_forbidden = list(map(lambda h: ROADTYPE_TO_ID[h], OD_FORBIDDEN))
        edges["allow_od"] = ~edges["road_type"].isin(od_forbidden)

        # Set speed of edges to default speed if NA.
        edges["urban"] = [urban_area.contains(geom) for geom in edges.geometry]
        urban_speeds = pd.DataFrame(
            list(DEFAULT_SPEED_URBAN.values()),
            index=list(DEFAULT_SPEED_URBAN.keys()),
            columns=["urban_speed"],
        )
        rural_speeds = pd.DataFrame(
            list(DEFAULT_SPEED_RURAL.values()),
            index=list(DEFAULT_SPEED_RURAL.keys()),
            columns=["rural_speed"],
        )
        default_speeds = pd.concat((urban_speeds, rural_speeds), axis=1)
        default_speeds.index = default_speeds.index.map(lambda rt: ROADTYPE_TO_ID[rt])
        edges = edges.merge(default_speeds, left_on="road_type", right_index=True, how="left")
        edges.loc[edges["speed"].isna() & edges["urban"], "speed"] = edges["urban_speed"]
        edges.loc[edges["speed"].isna() & ~edges["urban"], "speed"] = edges["rural_speed"]

        # Removing duplicate edges.
        st_count = edges.value_counts(subset=["source", "target"])
        to_remove = list()
        for s, t in st_count.loc[st_count > 1].index:
            dupl = edges.loc[(edges["source"] == s) & (edges["target"] == t)]
            # Keep in priority (i) main graph edge, (ii) largest capacity edge, (iii) smallest
            # free-flow travel time edge.
            nb_mains = dupl["main_graph"].sum()
            nb_secondary = len(dupl) - nb_mains
            indices = list()
            if nb_mains == 1:
                indices.extend(dupl.loc[~dupl["main_graph"]].index)
            else:
                if nb_mains > 0 and nb_secondary > 0:
                    indices.extend(dupl.loc[~dupl["main_graph"]].index)
                    dupl = dupl.loc[dupl["main_graph"]].copy()
                max_capacity = dupl["capacity"].max()
                indices.extend(dupl.loc[dupl["capacity"] < max_capacity].index)
                dupl = dupl.loc[dupl["capacity"] == max_capacity].copy()
                if len(dupl) > 1:
                    tt = dupl["length"] / (dupl["speed"] / 3.6)
                    id_min = tt.index[tt.argmin()]
                    indices.extend(dupl.loc[dupl.index != id_min].index)
            to_remove.extend(indices)
        if to_remove:
            print("Warning. Removing {} duplicate edges.".format(len(to_remove)))
            edges.drop(labels=to_remove, inplace=True)

        # Add a column for the indices of the edges (in the full network).
        edges.reset_index(inplace=True, drop=True)
        edges["index"] = edges.index

        print("Number of edges: {}".format(len(edges)))
        print("Number of edges in main graph: {}".format(edges["main_graph"].sum()))

        edges = edges[
            [
                "geometry",
                "index",
                "source",
                "target",
                "length",
                "speed",
                "lanes",
                "main_graph",
                "allow_od",
                "capacity",
                "osm_id",
                "name",
                "road_type",
            ]
        ]

        self.edges_df = edges

    def write_edges(self, filename):
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.edges_df.to_file(filename, driver="FlatGeobuf")


def buffer(geom, distance):
    wgs84 = pyproj.CRS("EPSG:4326")
    metric_crs = pyproj.CRS(METRIC_CRS)
    project = pyproj.Transformer.from_crs(wgs84, metric_crs, always_xy=True).transform
    inverse_project = pyproj.Transformer.from_crs(metric_crs, wgs84, always_xy=True).transform
    metric_geom = transform(project, geom)
    buffered_geom = metric_geom.buffer(distance).simplify(0, preserve_topology=False)
    geom = transform(inverse_project, buffered_geom)
    return geom


if __name__ == "__main__":

    t0 = time.time()

    # File does not exists or is not in the same folder as the script.
    if not os.path.exists(OSM_FILE):
        print("File not found: {}".format(OSM_FILE))
        sys.exit(0)

    print("Finding nodes...")
    node_reader = NodeReader()
    node_reader.apply_file(OSM_FILE, locations=True, idx="flex_mem")

    print("Reading edges...")
    edge_reader = EdgeReader(node_reader.nodes)
    edge_reader.apply_file(OSM_FILE, locations=True, idx="flex_mem")

    print("Finding urban areas...")
    area_reader = UrbanAreasReader()
    area_reader.apply_file(OSM_FILE, locations=True, idx="flex_mem")
    urban_area = area_reader.get_urban_area()

    # Buffer the urban areas by 50 meters to capture all nearby roads.
    urban_area = buffer(urban_area, 50)
    urban_area = prep(urban_area)

    print("Post-processing...")
    edge_reader.post_process(urban_area)

    print("Writing edges...")
    edge_reader.write_edges(EDGE_FILE)

    print("Done!")

    print("Total running time: {:.2f} seconds".format(time.time() - t0))
