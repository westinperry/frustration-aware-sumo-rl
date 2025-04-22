#!/usr/bin/env python3
import os
import random
import numpy as np
import sys

def generate_routefile(output_path,
                       max_steps=1000,
                       vehs_per_hour=600,
                       peds_per_hour=600,
                       seed=42):
    random.seed(seed)
    np.random.seed(seed)

    veh_rate = vehs_per_hour / 3600.0
    ped_rate = peds_per_hour / 3600.0

    # Header with CALM pedestrians
    header = [
        "<routes>",
        '  <vType id="car" accel="1.0" decel="4.5" maxSpeed="25" length="5"/>',
        '  <personType id="pedestrian" vClass="pedestrian" speed="1.0" impatience="0.0" jmCrossingGap="10.0" jmTimeGap="999"/>',
    ]

    trips = []

    # VEHICLES: map incoming edges to allowed outgoing edges
    valid_routes = {
        "N2TL": ["TL2E", "TL2S", "TL2W"],
        "E2TL": ["TL2N", "TL2S", "TL2W"],
        "S2TL": ["TL2N", "TL2E", "TL2W"],
        "W2TL": ["TL2N", "TL2E", "TL2S"],
    }

    # Generate vehicle trips
    for vidx, (incoming, outgoings) in enumerate(valid_routes.items()):
        for t in range(max_steps):
            if random.random() < veh_rate:
                outgoing = random.choice(outgoings)
                block = [
                    f'  <vehicle id="veh_{vidx}_{t}" type="car" depart="{t}" departLane="random" departSpeed="max">',
                    f'    <route edges="{incoming} {outgoing}"/>',
                    "  </vehicle>",
                ]
                trips.append((t, block))

    # --- PEDESTRIAN ROUTES TO UNCOMMENT AND TEST ONE BY ONE ---
    ped_crossings = []
    # North to South (single crossing)
    ped_crossings.append((":DN_w0", "N2TL", ":TL_w0", ":TL_c0", ":TL_w1", "TL2E"))
    # East to West (single crossing)
    ped_crossings.append((":DE_w0", "E2TL", ":TL_w1", ":TL_c1", ":TL_w2", "TL2S"))
    # South to North (single crossing)
    ped_crossings.append((":DS_w0", "S2TL", ":TL_w2", ":TL_c2", ":TL_w3", "TL2W"))
    # West to East (single crossing)
    ped_crossings.append((":DW_w0", "W2TL", ":TL_w3", ":TL_c3", ":TL_w0", "TL2N"))

    # Generate pedestrian trips
    for pidx, path in enumerate(ped_crossings):
        edges_str = " ".join(path)
        for t in range(max_steps):
            if random.random() < ped_rate:
                block = [
                    f'  <person id="ped_{pidx}_{t}" personType="pedestrian" depart="{t}" departPos="0">',
                    f'    <walk edges="{edges_str}"/>',
                    "  </person>",
                ]
                trips.append((t, block))

    # Sort by depart time
    trips.sort(key=lambda x: x[0])

    # Write to file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for line in header:
            f.write(line + "\n")
        for _, block in trips:
            for line in block:
                f.write(line + "\n")
        f.write("</routes>\n")

if __name__ == "__main__":
    generate_routefile("intersection/episode_routes.rou.xml")
