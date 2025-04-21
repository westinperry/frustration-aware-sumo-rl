# route_generator.py

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

    # Header
    header = [
        "<routes>",
        '  <vType id="car" accel="1.0" decel="4.5" maxSpeed="25" length="5"/>',
        '  <personType id="pedestrian" vClass="pedestrian" speed="1.0"/>',
    ]

    # Collect (depart_time, xml_block)
    trips = []

    # VEHICLES: map incoming edges to allowed outgoing edges (no U-turns)
    valid_routes = {
        "N2TL": ["TL2E", "TL2S", "TL2W"],
        "E2TL": ["TL2N", "TL2S", "TL2W"],
        "S2TL": ["TL2N", "TL2E", "TL2W"],
        "W2TL": ["TL2N", "TL2E", "TL2S"],
    }

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

    # PEDESTRIANS
    ped_crossings = [
        (":TL_w0", ":TL_w1", [":TL_c0"]),  # N→S
        (":TL_w2", ":TL_w3", [":TL_c1"]),  # E→W
        (":TL_w4", ":TL_w5", [":TL_c2"]),  # S→N
        (":TL_w6", ":TL_w7", [":TL_c3"]),  # W→E
    ]
    for pidx, (_from, _to, path) in enumerate(ped_crossings):
        edges_str = " ".join(path)
        if not edges_str:
            continue
        for t in range(max_steps):
            if random.random() < ped_rate:
                block = [
                    f'  <person id="ped_{pidx}_{t}" personType="pedestrian" depart="{t}">',
                    f'    <walk edges="{edges_str}"/>',
                    "  </person>",
                ]
                trips.append((t, block))

    # Sort by depart time
    trips.sort(key=lambda x: x[0])

    # Write out
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