import os

import torch
import network
import convert

abs_path = "/home/sergey/python/simulator_data_gen/nonperiodic_bimodal_3"
data = []
for t in os.listdir(abs_path):
    current_dir = os.path.join(abs_path, t, "network_data")
    # print(current_dir)
    for d in os.listdir(current_dir):
        local_dir = os.path.join(current_dir, d)
        print(local_dir)
        current_network = network.Network.from_data_file(
            os.path.join(local_dir, "network.lmp"),
            include_angles=True,
            include_dihedrals=False,
            include_default_masses=1e6)
        sim = convert.parse_dump(
            os.path.join(local_dir, "dump.lammpstrj"),
            current_network,
            node_features="coord",
            skip=1
            )
        data.append(sim)

# FOR OLD STUFF
# data = []
# for d in os.listdir(abs_path):
#     local_dir = os.path.join(abs_path, d)
#     print(local_dir)
#     current_network = network.Network.from_atoms(os.path.join(local_dir, "coord.dat"), include_angles=False, include_dihedrals=False, include_default_masses=100000.0)
#     sim = convert.parse_dump(os.path.join(local_dir, "dump.lammpstrj"), current_network, node_features="coord")
#     data.append(sim)

torch.save(data, "data_nonperiodic.pt")
