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
T = 10
abs_path = "/home/sergey/work/simulator_data_gen/one_over_l"
data = []
for size_dir in os.listdir(abs_path):
    if size_dir != 'data_generation.log' and int(size_dir.split('_')[0]) < 100:
        current_dir = os.path.join(abs_path, size_dir)
        for network_dir in os.listdir(current_dir):
            local_dir = os.path.join(current_dir, network_dir)
            for comp_dir in os.listdir(local_dir):
                if comp_dir.endswith(f"Tover{T}"):
                    deep_dir = os.path.join(local_dir, comp_dir)
                    print(deep_dir)
                    current_network = network.Network.from_data_file(
                        os.path.join(deep_dir, "network.lmp"),
                        include_angles=True,
                        include_dihedrals=False,
                        include_default_masses=1e6)
                    current_network.set_angle_coeff(0.00)
                    for bond in current_network.bonds:
                        bond.bond_coefficient = 1/bond.length
                    sim = convert.parse_dump(
                        os.path.join(deep_dir, "dump.lammpstrj"),
                        current_network,
                        node_features="coord",
                        skip=1
                        )
                    data.append(sim)

torch.save(data, f"/home/sergey/work/simulator_data_gen/validation_Tover{T}.pt")

# FOR OLD STUFF
# data = []
# for d in os.listdir(abs_path):
#     local_dir = os.path.join(abs_path, d)
#     print(local_dir)
#     current_network = network.Network.from_atoms(os.path.join(local_dir, "coord.dat"), include_angles=False, include_dihedrals=False, include_default_masses=100000.0)
#     sim = convert.parse_dump(os.path.join(local_dir, "dump.lammpstrj"), current_network, node_features="coord")
#     data.append(sim)
