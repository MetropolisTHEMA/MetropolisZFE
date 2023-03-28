import geopandas as gpd
from shapely.geometry import Polygon

# Path to the file where the A86 geometry is stored.
A86_FILE = "./output/ZFE/a86.geojson"
# Path to the file where the Métropole du Grand Paris geometry is stored.
MGP_FILE = "./output/ZFE/mgp.geojson"
# Path to the file where the ZFE geometry should be stored.
ZFE_FILE = "./output/ZFE/zfe.geojson"
# Path to the file where the edges are stored.
EDGES_FILE = "./output/osm_network/osm_edges.fgb"
EDGES_DRIVER = "FlatGeobuf"

print("Creating ZFE boundaries...")
# Read the A86 and MGP as Polygons.
a86 = gpd.read_file(A86_FILE).to_crs("epsg:2154")
mgp = gpd.read_file(MGP_FILE).to_crs("epsg:2154")

a86 = Polygon(a86.iloc[0]["geometry"])
mgp = Polygon(mgp.iloc[0]["geometry"])

# Reduce the A86 polygon by 100 meters to exclude the A86 itself from the ZFE.
a86 = a86.buffer(-100)

# The ZFE is the intersection of the A86 Polygon and the MGP Polygon.
zfe = a86 & mgp

s = gpd.GeoSeries(zfe)
s.name = "ZFE"
s.to_file(ZFE_FILE, driver="GeoJSON")

# Read edges.
print("Reading edges...")
edges = gpd.read_file(EDGES_FILE, driver=EDGES_DRIVER).to_crs("epsg:2154")

# Mark edges in the ZFE.
print("Finding edges inside the ZFE...")
edges["zfe"] = edges.geometry.within(zfe)

n = edges["zfe"].sum()
print("{} edges in the ZFE (representing {:.2%} of edges)".format(n, n / len(edges)))

print("Writing edges...")
edges.to_file(EDGES_FILE, driver=EDGES_DRIVER)
