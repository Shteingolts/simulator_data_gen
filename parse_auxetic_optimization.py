from multiprocessing import Pool
import os
from copy import deepcopy

import torch

import convert
import network
from lammps_scripts import CompressionSimulation, TemperatureRange
from main import (
    CalculationResult,
    CalculationSetup,
    ElasticData,
    StepResult,
    load_optimization_log,
    run_lammps,
)


def add_pruned_bonds_back(original_network: network.Network, pruned_network: network.Network):
    pruned_bonds = list(set(original_network.bonds) - set(pruned_network.bonds))
    for b in pruned_bonds:
        b.bond_coefficient = 0.0000

    with_dummies = deepcopy(pruned_network)
    combined_bonds = with_dummies.bonds + pruned_bonds
    with_dummies.bonds = combined_bonds
    with_dummies.header.bonds = len(combined_bonds)
    with_dummies.header.bond_types = len(combined_bonds)
    return with_dummies


def extract_from_opt(history, n_networks, add_pruned: bool) -> list[network.Network]:
    total_steps = len(history)
    step_size = total_steps // n_networks
    networks = [history[i].network for i in range(0, total_steps, step_size)]
    networks.append(history[-1].network)
    if add_pruned:
        networks = [
            add_pruned_bonds_back(history[0].network, networks[i])
            for i in range(len(networks))
        ]
    return networks


def extract_from_dump(
    path: str,
    include_angles: bool = True,
    node_features: str = "coord",
    skip: int = 1,
    network_filename: str = "network.lmp",
    dump_filename: str = "dump.lammpstrj",
):
    print(path)
    current_network = network.Network.from_data_file(
        os.path.join(path, network_filename),
        include_angles=include_angles,
        include_dihedrals=False,
    )

    sim = convert.parse_dump(
        os.path.join(path, dump_filename),
        current_network,
        node_features=node_features,
        skip=skip,
    )
    return sim


data_path = "/home/sergey/work/auxetic_optimizer/auxetic_data/"
comp_dir = "/home/sergey/python/simulator_data_gen/new_comp_lowT_noang_13.10.2024"
for network_size in os.listdir(data_path):
    local_dir = os.path.join(data_path, network_size)
    for subdir in os.listdir(local_dir):
        subdir_path = os.path.join(local_dir, subdir)
        history = load_optimization_log(os.path.join(subdir_path, "optimization_log.pkl"))
        networks = extract_from_opt(history, 6, add_pruned=True)

        for i, n in enumerate(networks):
            comp_sim = CompressionSimulation(
                box_size=n.box.x,
                network_filename="network.lmp",
                strain=0.03,
                temperature_range=TemperatureRange(1e-7, 1e-7, 10),
            )
            current_dir = os.path.join(comp_dir, str(network_size), str(subdir), str(i))
            print(current_dir)
            os.makedirs(current_dir, exist_ok=True)
            new = deepcopy(n)
            new.set_angle_coeff(0.00)
            new.write_to_file(os.path.join(current_dir, "network.lmp"))
            comp_sim.write_to_file(current_dir)
            run_lammps(current_dir, "in.deformation")

data = []
paths = []
for network_size in os.listdir(comp_dir):
    local_dir = os.path.join(comp_dir, str(network_size))
    for subdir in os.listdir(local_dir):
        subdir_path = os.path.join(local_dir, subdir)
        for ld in os.listdir(subdir_path):
            current_dir = os.path.join(subdir_path, ld)
            print(current_dir)
            paths.append(current_dir)
with Pool(12) as pool:
    data = pool.map(extract_from_dump, paths)

os.chdir("/home/sergey/python/simulator_data_gen/")
torch.save(data, "data_aux_opt_lowT_448sims_noang.pt")
