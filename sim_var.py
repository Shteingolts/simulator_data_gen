"""The same as `random_tries.ipynb`, but runnable from terminal.
"""
import os
import random
from multiprocessing import Pool

import numpy as np

from network import Bond, Network
from simulation import (
    CompressionSimulation,
    LJSimulation,
    TemperatureRange,
    construct_network,
    run_lammps_calc,
)


def prune_edges(network: Network, portion: float) -> Network:
    """Prunes a portion of bonds randomly in the network

    Parameters
    ----------
    network : Network
        
    portion : float
        Number of bonds to prune, 0 to 1

    Returns
    -------
    Network
        
    """
    
    bonds_to_prune: list[Bond] = random.sample(network.bonds, int(len(network.bonds)*portion))
    for bond in bonds_to_prune:
        bond.bond_coefficient = bond.bond_coefficient / 1000
    
    return network


def gen_bimodal_coeffs(n_edges: int, k: float = 0.3, mu1=0.0, sigma1=0.33, mu2=1.0, sigma2=0.18) -> list[float]:
    """Generates a bimodal distribution of numbers with a given parameters.

    Parameters
    ----------
    n_edges : int
        total number of entries in the distribution
    k : float, optional
        a portion of entries altered by the second distribution, by default 0.3
    mu1 : float, optional
        by default 0.0
    sigma1 : float, optional
        by default 0.33
    mu2 : float, optional
        by default 1.0
    sigma2 : float, optional
        by default 0.18

    Returns
    -------
    list[int]
        
    """
    
    # Populate the lower distribution
    lower: list[float] = []
    while len(lower) < int(k * n_edges):
        n = np.random.normal(mu1, sigma1)
        if 1.0 >= n > 0.01:
            lower.append(n)
        else:
            continue

    # Populate the higher distribution
    higher: list[float] = []
    while len(higher) < int((1-k) * n_edges):
        n = np.random.normal(mu2, sigma2)
        if 1.0 >= n > 0.01:
            higher.append(n)
        else:
            continue
    
    # Concat the two halves and make sure that the resuling length matches the number of entries
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


def prune_bimodal(network: Network) -> Network:
    n_edges = len(network.bonds)
    bimodal_coeffs = gen_bimodal_coeffs(n_edges)
    assert(len(bimodal_coeffs) == n_edges)

    for bond, k in zip(network.bonds, bimodal_coeffs):
        bond.bond_coefficient = bond.bond_coefficient * k
    
    return network


calculation_directory = os.path.join(os.getcwd(), "nonperiodic_bimodal_3")
print(f"Main dir: {calculation_directory}")

n_atoms = np.linspace(140, 240, 11, dtype=int)
print(f"N atoms:    {n_atoms}")
atom_types = np.linspace(3, 4, 4, dtype=int)
print(f"Atom types: {atom_types}")
atom_sizes = np.linspace(1.2, 1.8, 4, dtype=float)
print(f"Atom sizes: {atom_sizes}")
box_dim = [-40.0, 40.0, -40.0, 40.0, -0.1, 0.1]
print(f"Box size:   {box_dim}")
temperature_range = TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0)
print(f"Temp range: {temperature_range}")
n_steps = 30000
print(f"N steps:    {n_steps}")
batch_size = 5  # number of random networks with the same configuration
total_networks = len(n_atoms) * len(atom_types) * batch_size
print(f"N networks: {total_networks}")


def do_work(n_atoms: int):
    lj_sim = LJSimulation(
        n_atoms=n_atoms,
        n_atom_types=4,
        atom_sizes=atom_sizes,
        box_dim=box_dim,
        temperature_range=temperature_range,
        n_steps=n_steps,
    )
    comp_sim = CompressionSimulation(
        network_filename="network.lmp",  # do not change!
        strain=0.03,  # % of box X dimension
        strain_rate=1e-5,  # speed of compression
        desired_step_size=0.001,
        temperature_range=TemperatureRange(1e-7, 1e-7, 10.0),
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
        
        # Create a network from LJ coords. 
        # Carefull with beads mass, too low and everything breaks
        new_network = Network.from_atoms(
            os.path.join(target_dir, "coord.dat"),
            periodic=False,
            include_default_masses=5e8, # arbitrary mass for interesting compression
            include_angles=True,
            include_dihedrals=False
        )

        # Apply pruning based on bimodal distribution
        new_network = prune_bimodal(new_network)

        # Set angle coeffs to 0.01 for all angles
        new_network.set_angle_coeff(0.01)

        # Save the network into a file
        new_network.write_to_file(os.path.join(target_dir, "network.lmp"))

        # Run compression simulation
        comp_sim._recalc_dump_freq(new_network.box.x)
        comp_sim.write_to_file(target_dir)
        run_lammps_calc(
            target_dir,
            input_file="in.deformation",
            mode="single",
            num_threads=1,
            num_procs=1,
        )

if __name__ == "__main__":
    with Pool(10) as p:
        p.map(do_work, n_atoms)