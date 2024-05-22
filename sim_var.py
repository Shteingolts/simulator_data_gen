"""The same as `random_tries.ipynb`, but runnable from terminal.
"""
import os
import random
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.data import Data

from convert import assemble_data, network_from_data
from network import Network
from simulation import (
    CompressionSimulation,
    LJSimulation,
    TemperatureRange,
    run_lammps_calc,
)
from utils import get_periodic_estimation


def prune_edges(network: Network, portion: float):
    affected = random.sample(network.bonds, int(len(network.bonds)*portion))
    for bond in affected:
        bond.bond_coefficient = bond.bond_coefficient / 1000
    
    return network

def gen_k(n_edges: int, k: float = 0.3, mu1=0.0, sigma1=0.33, mu2=1.0, sigma2=0.18):
    lower = []
    while len(lower) < int(k * n_edges):
        n = np.random.normal(mu1, sigma1)
        if 1.0 >= n > 0.01:
            lower.append(n)
        else:
            continue

    higher = []
    while len(higher) < int((1-k) * n_edges):
        n = np.random.normal(mu2, sigma2)
        if 1.0 >= n > 0.01:
            higher.append(n)
        else:
            continue
    result = lower + higher
    if len(result) < n_edges:
        for _ in range(n_edges - len(result)):
            result.append(1.0)
    elif len(result) > n_edges:
        for _ in range(len(result) - n_edges):
            result.pop()
    else:
        pass
    return result

def change_edges(network: Network) -> Network:
    n_edges = len(network.bonds)
    ks = gen_k(n_edges)
    assert(len(ks) == n_edges)

    for bond, k in zip(network.bonds, ks):
        bond.bond_coefficient = bond.bond_coefficient * k
    
    return network

calculation_directory = os.path.join(os.getcwd(), "mptry")
print(f"Main dir: {calculation_directory}")

n_atoms = np.linspace(140, 240, 6, dtype=int)
print(f"N atoms:    {n_atoms}")
atom_types = np.linspace(3, 4, 2, dtype=int)
print(f"Atom types: {atom_types}")
atom_sizes = np.linspace(1.2, 1.8, 4, dtype=float)
print(f"Atom sizes: {atom_sizes}")
box_dim = [-7.0, 7.0, -7.0, 7.0, -0.1, 0.1]
print(f"Box size:   {box_dim}")
temperature_range = TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0)
print(f"Temp range: {temperature_range}")
n_steps = 30000
print(f"N steps:    {n_steps}")
batch_size = 5  # number of random networks with the same configuration
total_networks = len(n_atoms) * len(atom_types) * batch_size
print(f"N networks: {total_networks}")

# for n in n_atoms:
#     for n_types in atom_types:
#         lj_sim = LJSimulation(
#             n_atoms=n,
#             n_atom_types=n_types,
#             atom_sizes=atom_sizes[0:n_types],
#             box_dim=box_dim,
#             temperature_range=temperature_range,
#             n_steps=n_steps,
#         )
#         comp_sim = CompressionSimulation(
#             network_filename="network.lmp",  # do not change!
#             strain=0.03,  # % of box X dimension
#             strain_rate=1e-5,  # speed of compression
#             temperature_range=TemperatureRange(1e-7, 1e-7, 10.0),
#             dump_frequency=1000,  # `None` if you want 2000 steps or put a value to dump every N steps
#         )
#         custom_dir = os.path.join(calculation_directory, f"{n}_{n_types}")
#         os.makedirs(custom_dir)
        
#         assert(os.path.exists(custom_dir) and os.path.isdir(custom_dir))
#         data_dir = os.path.join(custom_dir, "network_data")
#         print(f"Data dir: {data_dir}")

#         # Create a separate directory for each network
#         for b in range(batch_size):
#             os.makedirs(os.path.join(data_dir, str(b + 1)))
#         dirs = os.listdir(data_dir)
#         dirs.sort(key=lambda x: int(x))

#         # Work with each network one by one
#         for network_dir in dirs:
#             print(f"Network dir: {network_dir}")
#             target_dir = os.path.join(data_dir, network_dir)
#             print(f"Target dir: {target_dir}")

#             lj_sim.write_to_file(target_dir)
#             run_lammps_calc(target_dir, input_file="lammps.in", mode="single")
            
#             # carefull with beads mass, too low and everything breaks
#             new_network = Network.from_atoms(
#                 os.path.join(target_dir, "coord.dat"),
#                 include_angles=False,
#                 include_dihedrals=False,
#                 include_default_masses=100000.0,
#                 periodic=True,
#             )
#             # =========================

#             graph = assemble_data(new_network.atoms, new_network.bonds, new_network.box, node_features="coord")
#             graph.x = graph.x + np.interp(
#                 torch.rand_like(graph.x),
#                 (
#                     torch.rand_like(graph.x).min(),
#                     torch.rand_like(graph.x).max()
#                 ),
#                 (-0.02*graph.box.x, 0.02*graph.box.x))
#             new_edges = get_periodic_estimation(graph, graph.box)
#             graph.edge_attr = new_edges

#             rand_network = network_from_data(graph)
#             rand_network.write_to_file("network.lmp")

#             #==========================
#             comp_sim.write_to_file(target_dir)
#             run_lammps_calc(
#                 target_dir,
#                 input_file="in.deformation",
#                 mode="mpi",
#                 num_threads=2,
#                 num_procs=2,
#             )

def do_work(n_atoms: int):
    lj_sim = LJSimulation(
            n_atoms=n_atoms,
            n_atom_types=4,
            atom_sizes=atom_sizes[0:4],
            box_dim=box_dim,
            temperature_range=temperature_range,
            n_steps=n_steps,
        )
    comp_sim = CompressionSimulation(
        network_filename="network.lmp",  # do not change!
        strain=0.03,  # % of box X dimension
        strain_rate=1e-5,  # speed of compression
        temperature_range=TemperatureRange(1e-7, 1e-7, 10.0),
        dump_frequency=1000,  # `None` if you want 2000 steps or put a value to dump every N steps
    )
    custom_dir = os.path.join(calculation_directory, f"{n_atoms}_{4}")
    os.makedirs(custom_dir)
    
    assert(os.path.exists(custom_dir) and os.path.isdir(custom_dir))
    data_dir = os.path.join(custom_dir, "network_data")
    print(f"Data dir: {data_dir}")

    # Create a separate directory for each network
    for b in range(batch_size):
        os.makedirs(os.path.join(data_dir, str(b + 1)))
    dirs = os.listdir(data_dir)
    dirs.sort(key=lambda x: int(x))

    # Work with each network one by one
    for network_dir in dirs:
        print(f"Network dir: {network_dir}")
        target_dir = os.path.join(data_dir, network_dir)
        print(f"Target dir: {target_dir}")

        lj_sim.write_to_file(target_dir)
        run_lammps_calc(target_dir, input_file="lammps.in", mode="single")
        
        # carefull with beads mass, too low and everything breaks
        new_network = Network.from_atoms(
            os.path.join(target_dir, "coord.dat"),
            include_angles=False,
            include_dihedrals=False,
            include_default_masses=100000.0,
            periodic=True,
        )
        # =========================

        graph = assemble_data(new_network.atoms, new_network.bonds, new_network.box, node_features="coord")
        graph.x = graph.x + np.interp(
            torch.rand_like(graph.x),
            (
                torch.rand_like(graph.x).min(),
                torch.rand_like(graph.x).max()
            ),
            (-0.02*graph.box.x, 0.02*graph.box.x))
        new_edges = get_periodic_estimation(graph, graph.box)
        graph.edge_attr = new_edges

        rand_network = network_from_data(graph)
        rand_network.write_to_file("network.lmp")

        #==========================
        comp_sim.write_to_file(target_dir)
        run_lammps_calc(
            target_dir,
            input_file="in.deformation",
            mode="single",
            num_threads=1,
            num_procs=1,
        )

if __name__ == "__main__":
    with Pool(8) as p:
        p.map(do_work, n_atoms)