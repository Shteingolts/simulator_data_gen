"""The same as `random_tries.ipynb`, but runnable from terminal.
"""
import os

import numpy as np

from simulation import (
    CompressionSimulation,
    LJSimulation,
    TemperatureRange,
    gen_sim_data,
)

calc_dir = "/home/sergey/python/simulator_data_gen/network_rand" # work

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
strains = [0.04, 0.06, 0.08]
n_steps = 30000
print(f"N steps:    {n_steps}")
batch_size = 7  # number of random networks with the same configuration
total_networks = len(n_atoms) * len(atom_types) * len(strains) * batch_size
print(f"N networks: {total_networks}")

for n in n_atoms:
    for n_types in atom_types:
        for strain in strains:
            ljsim = LJSimulation(
                n_atoms=n,
                n_atom_types=n_types,
                atom_sizes=atom_sizes[0:n_types],
                box_dim=box_dim,
                temperature_range=temperature_range,
                n_steps=n_steps,
            )
            comp_sim = CompressionSimulation(
                network_filename="network.lmp",  # do not change!
                strain=strain,  # % of box X dimension
                strain_rate=1e-5,  # speed of compression
                temperature_range=TemperatureRange(1e-7, 1e-7, 10.0),
                dump_frequency=None,  # `None` if you want 2000 steps or put a value to dump every N steps
            )
            custom_dir = os.path.join(calc_dir, f"{n}_{n_types}_{strain}")
            os.makedirs(custom_dir)
            gen_sim_data(
                custom_dir=custom_dir,
                lj_sim=ljsim,
                comp_sim=comp_sim,
                n_networks=batch_size,
            )
