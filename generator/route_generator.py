#!/usr/bin/env python3
import os
import random
import numpy as np
import sys
import matplotlib.pyplot as plt  # For plotting

def generate_beta_skewed_pedestrian_times(ped_count, max_steps, a=2.0, b=5.0, seed=42):
    np.random.seed(seed)
    raw_samples = np.random.beta(a, b, size=ped_count)
    ped_times = (raw_samples * (max_steps - 1)).astype(int)
    ped_times.sort()
    return ped_times

def generate_uniform_vehicle_times(veh_count, max_steps):
    veh_times = np.linspace(0, max_steps - 1, veh_count, dtype=int)
    return veh_times

def plot_departure_histograms(veh_times, ped_times, max_steps):
    plt.figure(figsize=(10, 5))
    plt.hist(veh_times, bins=50, alpha=0.7, label='Vehicles', color='blue', edgecolor='black')
    plt.hist(ped_times, bins=50, alpha=0.7, label='Pedestrians', color='orange', edgecolor='black')
    plt.title('Histogram of Departure Times')
    plt.xlabel('Simulation Step')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def generate_routefile(output_path,
                       max_steps=1000,
                       vehs_per_hour=600,
                       peds_per_hour=300,
                       seed=42,
                       plot=False):
    random.seed(seed)
    np.random.seed(seed)

    sim_hours = max_steps / 3600.0
    veh_count = int(round(vehs_per_hour * sim_hours))
    ped_count = int(round(peds_per_hour * sim_hours))

    header = [
        "<routes>",
        '  <vType id="car" accel="1.0" decel="4.5" maxSpeed="25" length="5"/>',
        '  <personType id="pedestrian" vClass="pedestrian" speed="1.0" impatience="0.0" jmCrossingGap="10.0" jmTimeGap="999"/>',
    ]

    trips = []

    # VEHICLES
    valid_routes = {
        "N2TL": ["TL2E", "TL2S", "TL2W"],
        "E2TL": ["TL2N", "TL2S", "TL2W"],
        "S2TL": ["TL2N", "TL2E", "TL2W"],
        "W2TL": ["TL2N", "TL2E", "TL2S"],
    }
    vehicle_edges = list(valid_routes.keys())
    num_veh_edges = len(vehicle_edges)
    veh_times = generate_uniform_vehicle_times(veh_count, max_steps)

    for i, t in enumerate(veh_times):
        incoming = vehicle_edges[i % num_veh_edges]
        outgoing = random.choice(valid_routes[incoming])
        block = [
            f'  <vehicle id="veh_{i}_{t}" type="car" depart="{t}" departLane="random" departSpeed="max">',
            f'    <route edges="{incoming} {outgoing}"/>',
            "  </vehicle>",
        ]
        trips.append((t, block))

    # PEDESTRIANS
    ped_crossings = [
        (":DN_w0", "N2TL", ":TL_w0", ":TL_c0", ":TL_w1", "TL2E"),
        (":DE_w0", "E2TL", ":TL_w1", ":TL_c1", ":TL_w2", "TL2S"),
        (":DS_w0", "S2TL", ":TL_w2", ":TL_c2", ":TL_w3", "TL2W"),
        (":DW_w0", "W2TL", ":TL_w3", ":TL_c3", ":TL_w0", "TL2N"),
    ]
    ped_times = generate_beta_skewed_pedestrian_times(ped_count, max_steps)

    for i, t in enumerate(ped_times):
        path = ped_crossings[i % len(ped_crossings)]
        edges_str = " ".join(path)
        block = [
            f'  <person id="ped_{i}_{t}" personType="pedestrian" depart="{t}" color="1,1,0" guiShape="pedestrian">',
            f'    <walk edges="{edges_str}"/>',
            "  </person>",
        ]
        trips.append((t, block))

    if plot:
        plot_departure_histograms(veh_times, ped_times, max_steps)

    # Sort and write to XML
    trips.sort(key=lambda x: x[0])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for line in header:
            f.write(line + "\n")
        for _, block in trips:
            for line in block:
                f.write(line + "\n")
        f.write("</routes>\n")

    print(f"Route file written to {output_path}")
    print(f"Generated {veh_count} vehicles and {ped_count} pedestrians")
    print(f"  • Vehicle spacing: ~{max_steps/veh_count:.1f} steps")
    print(f"  • Pedestrian distribution: Beta(2,5)")

if __name__ == "__main__":
    # Enable plot=True to visualize both histograms
    generate_routefile("intersection/episode_routes.rou.xml", plot=True)
