"""The same as `random_tries.ipynb`, but runnable from terminal.
"""
import argparse
import logging
import os
import random
from multiprocessing import Pool

import numpy as np

from network import Bond, Network
from simulation import (
    CompressionSimulation,
    LJSimulation,
    TemperatureRange,
    run_lammps_calc,
)

from utils import inject_noise

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


def randomize_LJ(n_atoms: int):
    atom_types = random.randint(2, 8)
    atom_sizes = [random.uniform(1.0, 2.0) for i in range(atom_types)]
    
    lj_sim = LJSimulation(
        n_atoms=n_atoms,
        n_atom_types=4,
        atom_sizes=[1.6, 1.4, 1.2, 1.0],
        box_dim=[-40, 40, -40, 40, -0.1, 0.1],
        temperature_range=TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0),
    )


def run_one_calc(
    local_calc_dir: str,
    n_atoms: int,
    masses: int,
    angle_coeff: float,
    pruning: str,
    pruning_parameters: str,
    noise: str,
    strain: float,
    strain_direction: str = 'x',
):
    os.makedirs(local_calc_dir)
    assert(os.path.exists(local_calc_dir) and os.path.isdir(local_calc_dir))
    print(local_calc_dir)

    lj_sim = LJSimulation(
        n_atoms=n_atoms,
        n_atom_types=4,
        atom_sizes=atom_sizes,
        box_dim=box_dim,
        temperature_range=temperature_range,
        n_steps=n_steps,
    )

    # LJ sim
    lj_sim.write_to_file(local_calc_dir)
    run_lammps_calc(local_calc_dir, input_file="lammps.in", mode="single")
        
    # Create a network from LJ coords. 
    new_network = Network.from_atoms(
        os.path.join(local_calc_dir, "coord.dat"),
        periodic=True,
        include_default_masses=masses, # arbitrary mass for interesting compression
        include_angles=True,
        include_dihedrals=False
    )

    # Inject noise into atom cordinates
    match noise:
        case 'yes':
            new_network = inject_noise(new_network, angle_coeffs=angle_coeff)
        case 'no':
            pass
        case _:
            raise Exception(f'Acceptable values for `noise` argument are: `yes` or `no`, got {noise} instead.')

    # Apply pruning based on bimodal distribution
    match pruning:
        case 'no':
            pass
        case 'random':
            try:
                start, end = map(int, pruning_parameters.split('-'))
            except Exception:
                raise TypeError("Could not parse your prunings parameters (-pp) into two ints. Try 'n-m'.")
            assert(isinstance(start, int))
            assert(isinstance(end, int))
            # print(start, end)
            new_network = prune_edges(new_network, random.randint(start, end)/100)
        case _:
            raise NotImplementedError("choice of pruning is not implemented yet")

    # Set angle coeffs for all angles
    new_network.set_angle_coeff(angle_coeff)


    # Save the network into a file
    for bond in new_network.bonds:
        bond.bond_coefficient = bond.bond_coefficient * 10
    new_network.write_to_file(os.path.join(local_calc_dir, "network.lmp"))

    # Run compression simulation
    comp_sim = CompressionSimulation(
        network_filename="network.lmp",  # do not change!
        strain=strain,  # % of box X dimension
        strain_rate=1e-5,  # speed of compression
        strain_direction=strain_direction,
        desired_step_size=0.001,
        temperature_range=TemperatureRange(1e-7, 1e-7, 10.0),
    )

    comp_sim._recalc_dump_freq(new_network.box.x)
    comp_sim.write_to_file(local_calc_dir)
    run_lammps_calc(
        local_calc_dir,
        input_file="in.deformation",
        mode="single",
        num_threads=1,
        num_procs=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument(
        "-min",
        "--min_size",
        type=int,
        default=200,
        help="min network size. Default is 200."
    )
    parser.add_argument(
        "-max",
        "--max_size",
        type=int,
        default=400,
        help="max network size. Default is 400."
    )
    parser.add_argument(
        "--n_sizes",
        type=int,
        default=11,
        help="number of network sizes in the given range. Default is 11."
    )
    parser.add_argument(
        "-batch",
        "--batch_size",
        type=int,
        default=10,
        help="number of networks of a given size. Default is 10."
    )
    parser.add_argument(
        "-s",
        "--strain",
        type=float,
        default=0.03,
        help="compression strain. Default is 0.03."
    )
    parser.add_argument(
        "-sr",
        "--strain_rate",
        type=float,
        default=1e-5,
        help="LAMMPS strain rate. Default is 1e-5."
    )
    parser.add_argument(
        "-ss",
        "--dump_strain_step_size",
        type=float,
        default=0.001,
        help="Difference between box sizes in real units between snapshots. Default is 0.001."
    )
    parser.add_argument(
        "-m",
        "--masses",
        type=float,
        help="node masses"
    )
    parser.add_argument(
        "-ang",
        "--angles",
        type=float,
        help="angle coeffitients"
    )
    parser.add_argument(
        "-p",
        "--pruning",
        type=str,
        help="type of edge pruning"
    )
    parser.add_argument(
        "-pp",
        "--pruning_parameters",
        type=str,
        help="two integers between 0 and 100, for example `1-40`, for random. Leave empty for bimodal and no pruning."
    )
    parser.add_argument(
        "-n",
        "--noise",
        type=str,
        help="Inject noise into networks to make them more diverse. By default each atom is displaced in x and y by a random value sampled from a gaussian with mu=0 and std=0.2*ave_bond_length"
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=10,
        help="number of cores to run on. Default is 10."
    )
    parser.add_argument(
        "-out",
        "--output_directory",
        type=str,
        help="name of the output data directory."
    )
    
    args = parser.parse_args()

    min_size = args.min_size
    max_size = args.max_size
    n_sizes = args.n_sizes
    batch_size = args.batch_size
    strain = args.strain
    strain_rate = args.strain_rate
    strain_rate = args.strain_rate
    dump_strain_step_size = args.dump_strain_step_size
    compression_temperature_range = TemperatureRange(1e-7, 1e-7, 10.0)
    mass = args.masses
    angle_coeffs = args.angles
    pruning = args.pruning
    pruning_parameters = args.pruning_parameters
    add_noise = args.noise
    cores = args.cores
    output_directory = args.output_directory

    os.makedirs(os.path.join(os.getcwd(), output_directory), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_directory, "data_generation.log"),
        level=logging.INFO,
        format='%(message)s'
    )

    calculation_directory = os.path.join(os.getcwd(), output_directory)
    n_atoms = np.linspace(min_size, max_size, n_sizes, dtype=int)
    atom_types = np.linspace(3, 4, 4, dtype=int)
    atom_sizes = np.linspace(1.2, 1.8, 4, dtype=float)
    box_dim = [-40.0, 40.0, -40.0, 40.0, -0.1, 0.1]
    temperature_range = TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0)
    n_steps = 30000
    batch_size = batch_size  # number of random networks with the same configuration
    total_sims = n_sizes * batch_size

    logging.info(f"Calculation directory: \n{calculation_directory}")
    logging.info(f"N cores:        {cores}")
    logging.info(f"Batch size:     {batch_size}")
    logging.info(f"Total sims:     {total_sims}")

    logging.info("\nLJ parameters:")
    logging.info(f"Atoms:       {n_atoms}")
    logging.info(f"Atom types:  {atom_types}")
    logging.info(f"Atom sizes:  {atom_sizes}")
    logging.info(f"Start box:   {box_dim}")
    logging.info(f"T range:     {temperature_range}")
    logging.info(f"Sim steps:   {n_steps}")

    logging.info("\nCompression parameters:")
    logging.info(f"Strain:         {strain}")
    logging.info(f"Strain rate:    {strain_rate}")
    logging.info(f"Dump step size: {dump_strain_step_size}")
    logging.info(f"T range:        {compression_temperature_range}")
    logging.info(f"Mass:           {mass:e}")
    logging.info(f"Angle coeffs:   {angle_coeffs}")
    logging.info(f"Pruning type:   {pruning}")
    logging.info(f"Pruning params: {pruning_parameters}")
    logging.info(f"Injects noise?: {add_noise}")

    # make paths
    paths = []
    atoms = []
    masses = []
    angles = []
    prunings = []
    pps = []
    noises = []
    strains = []
    strain_dirs = []
    for size in n_atoms:
        for i in range(batch_size):
            atoms.append(size)
            masses.append(mass)
            angles.append(angle_coeffs)
            prunings.append(pruning)
            pps.append(pruning_parameters)
            noises.append(add_noise)
            strains.append(strain)
            strain_dirs.append('x')
            paths.append(os.path.join(calculation_directory, f"{size}_{4}", "network_data", str(i+1)))
    
    inputs = list(zip(paths, atoms, masses, angles, prunings, pps, noises, strains, strain_dirs))

    with Pool(cores) as p:
        p.starmap(run_one_calc, inputs)