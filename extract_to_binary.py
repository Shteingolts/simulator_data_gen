import os

import torch
import network
import convert
from multiprocessing import Pool

def extract_from_dump(path: str, include_angles: bool = True, node_features: str = "coord", skip: int = 1):
    print(path)
    current_network = network.Network.from_data_file(
        os.path.join(path, "network.lmp"),
        include_angles=include_angles,
        include_dihedrals=False)
    sim = convert.parse_dump(
        os.path.join(path, "dump.lammpstrj"),
        current_network,
        node_features=node_features,
        skip=skip
        )
    return sim

raw_data_path = "/home/sergey/work/data_big_pruned_0.3_a0.01"
binary_data_path = "/home/sergey/work/data_big_pruned_0.3_a0.01.pt"
paths = []
for t in os.listdir(raw_data_path):
    current_dir = os.path.join(raw_data_path, t, "network_data")
    for d in os.listdir(current_dir):
        local_dir = os.path.join(current_dir, d)
        paths.append(local_dir)

if __name__ == "__main__":
    with Pool(12) as pool:
        data = pool.map(extract_from_dump, paths)
    torch.save(data, binary_data_path)
