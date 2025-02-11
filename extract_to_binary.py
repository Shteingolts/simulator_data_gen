import os

import torch
from torch_geometric.data import Data
import network
import convert
from multiprocessing import Pool
from utils import add_pruned_bonds_back


def extract_from_dump(path: str) -> list[Data]:
    print(path)
    current_network = network.Network.from_data_file(
        os.path.join(path, "network.lmp"),
        include_angles=True,
        include_dihedrals=False,
    )

    sim = convert.parse_dump(
        os.path.join(path, "dump.lammpstrj"),
        current_network,
        node_features='coord',
        skip=1,
    )
    return sim


# normal extract
raw_data_path = "/home/sergey/work/simulator_data_gen/stiff"
binary_data_path = "/home/sergey/work/simulator_data_gen/data/binary/stiff.pt"
paths = []
for t in os.listdir(raw_data_path):
    if t != "data_generation.log":
        current_dir = os.path.join(raw_data_path, t, "network_data")
        for d in os.listdir(current_dir):
            local_dir = os.path.join(current_dir, d)
            paths.append(local_dir)


# for auxetic
# raw_data_path = "/home/sergey/work/auxetic_optimizer/auxetic_data"
# binary_data_path = "/home/sergey/python/simulator_data_gen/auxetic_data_m1e6_ang0.01_noprun_with_missing_edges.pt"
# paths = []
# for t in os.listdir(raw_data_path):
#     current_dir = os.path.join(raw_data_path, t)
#     for d in os.listdir(current_dir):
#         local_dir = os.path.join(current_dir, d)
#         paths.append(os.path.join(local_dir, "compression"))

if __name__ == "__main__":
    with Pool() as pool:
        data = pool.map(extract_from_dump, paths)

    torch.save(data, binary_data_path)
