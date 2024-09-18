import os

import torch
from torch_geometric.data import Data
import network
import convert
from multiprocessing import Pool
from utils import add_pruned_bonds_back

def extract_from_dump(
    path: str,
    add_pruned_bonds: bool = True,
    include_angles: bool = True,
    node_features: str = "coord",
    skip: int = 1,
    network_filename: str = "final_result.lmp",
    dump_filename: str = "dump.lammpstrj"
) -> list[Data]:
    print(path)
    current_network = network.Network.from_data_file(
        os.path.join(path, network_filename),
        include_angles=include_angles,
        include_dihedrals=False)
    
    if add_pruned_bonds:
        original_network = network.Network.from_data_file(
            os.path.join(os.path.split(path)[0], "original_network.lmp"),
            include_dihedrals=False
        )
        current_network = add_pruned_bonds_back(original_network, current_network)

    sim = convert.parse_dump(
        os.path.join(path, dump_filename),
        current_network,
        node_features=node_features,
        skip=skip
        )
    return sim

# normal extract
# raw_data_path = "/home/sergey/work/auxetic_optimizer/auxetic_data"
# binary_data_path = "/home/sergey/work/auxetic_optimizer/auxetic_data_1e6_ang0.01_noprun.pt"
# paths = []
# for t in os.listdir(raw_data_path):
#     current_dir = os.path.join(raw_data_path, t, "network_data")
#     for d in os.listdir(current_dir):
#         local_dir = os.path.join(current_dir, d)
#         paths.append(local_dir)

# for auxetic 
raw_data_path = "/home/sergey/work/auxetic_optimizer/auxetic_data"
binary_data_path = "/home/sergey/python/simulator_data_gen/auxetic_data_m1e6_ang0.01_noprun_with_missing_edges.pt"
paths = []
for t in os.listdir(raw_data_path):
    current_dir = os.path.join(raw_data_path, t)
    for d in os.listdir(current_dir):
        local_dir = os.path.join(current_dir, d)
        paths.append(os.path.join(local_dir, "compression"))

if __name__ == "__main__":
    with Pool(12) as pool:
        data = pool.map(extract_from_dump, paths)
    torch.save(data, binary_data_path)
