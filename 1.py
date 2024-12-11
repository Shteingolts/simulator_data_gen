from multiprocessing import Pool
from os.path import join
from os import listdir
import network
from convert import parse_dump
import torch

raw_data_path = "/home/sergey/work/simulator_data_gen/even_more_noisy_pruned_data"
binary_data_path = "/home/sergey/work/simulator_data_gen/data/binary/even_more_data_part3.pt"
paths = []
for t in listdir(raw_data_path):
    if t != "data_generation.log" and 200 < int(t.split('_')[0]) < 255:
        current_dir = join(raw_data_path, t, "network_data")
        for d in listdir(current_dir):
            local_dir = join(current_dir, d)
            paths.append(local_dir)

def extract_from_dump(path: str):
    print(path)
    current_network = network.Network.from_data_file(
        join(path, "network.lmp"),
        include_angles=True,
        include_dihedrals=False,
    )

    sim = parse_dump(
        join(path, "dump.lammpstrj"),
        current_network,
        node_features='coord',
        skip=1,
    )
    return sim

with Pool() as pool:
    sims = pool.map(extract_from_dump, paths)

torch.save(sims, binary_data_path)