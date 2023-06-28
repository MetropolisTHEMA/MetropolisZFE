import os
import time
import json

import numpy as np
import pandas as pd
import geopandas as gpd

# Path to the FlatGeobuf file where edges are stored.
EDGE_FILE = "./output/osm_network/osm_edges.fgb"
# Path to the file where the trip data is stored.
TRIPS_FILE = "./output/trips/trips.csv"
# Use only trips whose mode is within the following modes
# (available values: car, car_passenger, pt, walk, bike).
MODES = ("car",)
# Path to the directory where the simulation input should be stored.
OUTPUT_DIR = "./output/runs/next_run/"
# Vehicle length in meters.
VEHICLE_LENGTH = 10.0 * 7.0
# Vehicle passenger-car equivalent.
VEHICLE_PCE = 10.0 * 1.0
# Period in which the departure time of the trip is chosen.
PERIOD = [3.0 * 3600.0, 10.0 * 3600.0]
# If True, enable bottlenecks using capacity defined by edges' variable `capacity`.
BOTTLENECK = True
# Value of time in the car, in euros / hour.
ALPHA = 15.0
# Value of arriving early at destination, in euros / hour.
BETA = 7.5
# Value of arriving late at destination, in euros / hour.
GAMMA = 30.0
# Time window for on-time arrival, in seconds.
DELTA = 0.0
# "Continuous": Continuous Logit.
# "Discrete": Multinomial Logit model (with fixed epsilons).
# "Exogenous": The departure time from synthetic population is used.
DEPARTURE_TIME_MODEL = "Discrete"
# Value of μ for the departure-time model (if DEPARTURE_TIME_MODEL is not "Exogenous").
DT_MU = 3.0
# Width of the departure-time choice intervals (for DEPARTURE_TIME_MODEL "Discrete" only).
DT_WIDTH = 5.0 * 60.0
# Bins of the departure-time choice intervals (for DEPARTURE_TIME_MODEL "Discrete" only).
DT_BINS = list(np.arange(PERIOD[0] + DT_WIDTH / 2, PERIOD[1], DT_WIDTH))
# How t* is computed given the observed arrival time.
def T_STAR_FUNC(ta):
    return ta


# If True, restrict polluting vehicles from entering the ZFE.
ZFE = False
# Crit'air labels which are forbidden in the ZFE.
INVALID_CRITAIRS = ("Crit'air 4", "Crit'air 5", "Inconnu", "Non classée")


# Seed for the random number generators.
SEED = 13081996
RNG = np.random.default_rng(SEED)
# Parameters to use for the simulation.
PARAMETERS = {
    "period": PERIOD,
    "init_iteration_counter": 1,
    "learning_model": {
        "type": "Exponential",
        "value": {
            "alpha": 0.99,
        },
    },
    "stopping_criteria": [
        {
            "type": "MaxIteration",
            "value": 200,
        },
    ],
    "update_ratio": 1.0,
    "random_seed": SEED,
    "network": {
        "road_network": {
            "recording_interval": 300.0,
            "spillback": False,
            "max_pending_duration": 30.0,
        }
    },
    "nb_threads": 0,  # default 0: uses all possible threads
}


def read_edges():
    print("Reading edges")
    edges = gpd.read_file(EDGE_FILE)
    edges = edges.loc[edges["main"]].copy()
    edges.sort_values("index_main", inplace=True)
    return edges


def read_trips():
    print("Reading trips")
    trips = pd.read_csv(TRIPS_FILE)
    trips = trips.loc[trips["mode"].isin(MODES)].copy()
    trips = trips.loc[
        (trips["departure_time"] >= PERIOD[0]) & (trips["arrival_time"] <= PERIOD[1])
    ].copy()
    print("{} trips read".format(len(trips)))
    return trips


def generate_road_network(edges):
    print("Creating Metropolis road network")
    metro_edges = list()
    for _, row in edges.iterrows():
        edge = [
            row["source"],
            row["target"],
            {
                "id": int(row["index_main"]),
                "base_speed": float(row["speed"]) / 3.6,
                "length": float(row["length"]),
                "lanes": int(row["lanes"]),
                "speed_density": {
                    "type": "FreeFlow",
                },
                "overtaking": True,
            },
        ]
        if capacity := row["capacity"]:
            edge[2]["bottleneck_flow"] = capacity / 3600.0
        #  if const_tt := CONST_TT.get(row["neighbor_count"]):
        #  edge[2]["constant_travel_time"] = const_tt
        metro_edges.append(edge)

    graph = {
        "edges": metro_edges,
    }

    vehicles = [
        {
            "length": VEHICLE_LENGTH,
            "pce": VEHICLE_PCE,
            "speed_function": {
                "type": "Base",
            },
        },
    ]

    if ZFE:
        # Add a second vehicle representing the restricted vehicles.
        zfe_edges = list(edges.loc[edges["zfe"], "index_main"])
        vehicles.append(
            {
                "length": VEHICLE_LENGTH,
                "pce": VEHICLE_PCE,
                "speed_function": {
                    "type": "Base",
                },
                "restricted_edges": zfe_edges,
            }
        )

    road_network = {
        "graph": graph,
        "vehicles": vehicles,
    }
    return road_network


def generate_agents(trips):
    trips.sort_values(["person_id", "trip_index"], inplace=True)
    trips["is_first"] = trips["trip_id"].isin(trips.groupby("person_id")["trip_id"].first())
    trips["is_last"] = trips["trip_id"].isin(trips.groupby("person_id")["trip_id"].last())
    # Fill values for virtual legs.
    trips["origin_delay"] = trips["origin_delay"].fillna(0)
    trips["destination_delay"] = trips["destination_delay"].fillna(0)
    # Compute stopping times.
    trips["next_departure_time"] = trips["departure_time"].shift(-1, fill_value=0)
    trips["stopping_time"] = trips["next_departure_time"] - trips["arrival_time"]
    trips.loc[trips["is_last"], "stopping_time"] = 0
    trips["next_origin_delay"] = trips["origin_delay"].shift(-1, fill_value=0)
    trips.loc[trips["is_last"], "next_origin_delay"] = 0
    print("Generating agents")
    random_u = iter(RNG.uniform(size=len(trips)))
    agents = list()
    nb_persons = trips["person_id"].nunique()
    origins = set()
    destinations = set()
    for i, (person_id, idx) in enumerate(trips.groupby("person_id").groups.items()):
        legs = list()
        for _, trip in trips.loc[idx].iterrows():
            print(f"person_id : {i + 1}/{nb_persons}", end="\r")

            t_star = T_STAR_FUNC(trip["arrival_time"]) - trip["destination_delay"]

            if trip["road_leg"]:
                origin = int(trip["O_node"])
                destination = int(trip["D_node"])
                origins.add(origin)
                destinations.add(destination)
                if ZFE and trip["is_feasible"] and trip["critair"] in INVALID_CRITAIRS:
                    vehicle_id = 1
                else:
                    vehicle_id = 0
                leg = {
                    "class": {
                        "type": "Road",
                        "value": {
                            "origin": origin,
                            "destination": destination,
                            "vehicle": vehicle_id,
                        },
                    }
                }
                leg["travel_utility"] = {
                    "type": "Polynomial",
                    "value": {
                        "b": -ALPHA / 3600.0,
                    },
                }
                leg["schedule_utility"] = {
                    "type": "AlphaBetaGamma",
                    "value": {
                        "beta": BETA / 3600.0,
                        "gamma": GAMMA / 3600.0,
                        "t_star_high": t_star + DELTA / 2.0,
                        "t_star_low": t_star - DELTA / 2.0,
                    },
                }
                leg["stopping_time"] = (
                    trip["destination_delay"] + trip["stopping_time"] + trip["next_origin_delay"]
                )
            else:
                leg = {"class": {"type": "Virtual", "value": trip["travel_time"]}}
                leg["stopping_time"] = trip["stopping_time"] + trip["next_origin_delay"]
            legs.append(leg)
        if DEPARTURE_TIME_MODEL == "Continuous":
            departure_time_model = {
                "type": "ContinuousChoice",
                "value": {
                    "period": PERIOD,
                    "choice_model": {
                        "type": "Logit",
                        "value": {
                            "u": next(random_u),
                            "mu": DT_MU,
                        },
                    },
                },
            }
        elif DEPARTURE_TIME_MODEL == "Discrete":
            departure_time_model = {
                "type": "DiscreteChoice",
                "value": {
                    "values": DT_BINS,
                    "choice_model": {
                        "type": "Deterministic",
                        "value": {
                            "u": RNG.uniform(),
                            "constants": list(RNG.gumbel(scale=DT_MU, size=len(DT_BINS))),
                        },
                    },
                    "offset": RNG.uniform(-DT_WIDTH / 2, DT_WIDTH / 2),
                },
            }
        else:
            departure_time_model = {
                "type": "Constant",
                "value": trips.loc[idx[0], "departure_time"],
            }
        car_mode = {
            "type": "Trip",
            "value": {
                "legs": legs,
                "origin_delay": trips.loc[idx[0], "origin_delay"],
                "departure_time_model": departure_time_model,
            },
        }
        agent = {
            "id": person_id,
            "modes": [car_mode],
        }
        agents.append(agent)
    print(
        "\nGenerated {} agents, with {} legs ({} being road legs)".format(
            len(agents), len(trips), trips["road_leg"].sum()
        )
    )
    print("Number of unique origins: {}".format(len(origins)))
    print("Number of unique destinations: {}".format(len(destinations)))
    return agents


if __name__ == "__main__":

    t0 = time.time()

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Writing parameters")
    with open(os.path.join(OUTPUT_DIR, "parameters.json"), "w") as f:
        f.write(json.dumps(PARAMETERS))

    edges = read_edges()

    trips = read_trips()

    agents = generate_agents(trips)
    del trips

    print("Writing agents")
    with open(os.path.join(OUTPUT_DIR, "agents.json"), "w") as f:
        f.write(json.dumps(agents))
    del agents

    road_network = generate_road_network(edges)
    del edges

    print("Writing road network")
    with open(os.path.join(OUTPUT_DIR, "network.json"), "w") as f:
        f.write(json.dumps(road_network))

    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
