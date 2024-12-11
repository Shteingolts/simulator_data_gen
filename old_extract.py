import os
import random

import numpy as np
import torch

import convert
from network import Bond, Network
import network
from simulation import (
    CompressionSimulation,
    LJSimulation,
    TemperatureRange,
    construct_network,
    run_lammps_calc,
)

abs_path = "/home/sergey/work/simulator_data_gen/data/raw/noised_prund"
data = []
for t in os.listdir(abs_path):
    if t != 'data_generation.log':
        current_dir = os.path.join(abs_path, t, "network_data")
        for d in os.listdir(current_dir):
            local_dir = os.path.join(current_dir, d)
            print(local_dir)
            current_network = network.Network.from_data_file(
                os.path.join(local_dir, "network.lmp"),
                include_angles=True,
                include_dihedrals=False,
                include_default_masses=1e6)
            current_network.set_angle_coeff(0.00)
            sim = convert.parse_dump(
                os.path.join(local_dir, "dump.lammpstrj"),
                current_network,
                node_features="coord",
                skip=1
                )
            data.append(sim)

torch.save(data, f"{os.path.basename(abs_path)}.pt")

# FOR OLD STUFF
# data = []
# for d in os.listdir(abs_path):
#     local_dir = os.path.join(abs_path, d)
#     print(local_dir)
#     current_network = network.Network.from_atoms(os.path.join(local_dir, "coord.dat"), include_angles=False, include_dihedrals=False, include_default_masses=100000.0)
#     sim = convert.parse_dump(os.path.join(local_dir, "dump.lammpstrj"), current_network, node_features="coord")
#     data.append(sim)
