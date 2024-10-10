"""The same as `random_tries.ipynb`, but runnable from terminal and multiproc.
"""
import os
from multiprocessing import Pool

import numpy as np
import torch
from torch_geometric.data import Data

from convert import assemble_data, network_from_data, parse_dump
from network import Box, Network
from simulation import (
    CompressionSimulation,
    LJSimulation,
    TemperatureRange,
    run_lammps_calc,
)
from utils import flatten, get_periodic_estimation


calculation_directory = os.path.join(os.getcwd(), "mptry")
print(f"Main dir: {calculation_directory}")

n_atoms = np.linspace(140, 240, 51, dtype=int)
print(f"N atoms:    {n_atoms}")
atom_types = np.linspace(3, 4, 2, dtype=int)
print(f"Atom types: {atom_types}")
atom_sizes = np.linspace(1.2, 1.8, 4, dtype=float)
print(f"Atom sizes: {atom_sizes}")
box_dim = [-7.0, 7.0, -7.0, 7.0, -0.1, 0.1]
print(f"Box size:   {box_dim}")
temperature_range = TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0)
print(f"Temp range: {temperature_range}")
N_STEPS = 30000
print(f"N steps:    {N_STEPS}")
BATCH_SIZE = 20  # number of random networks with the same configuration
total_networks = len(n_atoms) * len(atom_types) * BATCH_SIZE
print(f"N networks: {total_networks}")

class TestResult:
    original_p: float
    optimized_lammps_p: float
    optimized_sim_p: float
    path: str

    def __init__(self, original_p: float, optimized_lammps_p: float, optimized_sim_p: float, path: str) -> None:
        self.original_p = original_p
        self.optimized_lammps_p = optimized_lammps_p
        self.optimized_sim_p = optimized_sim_p
        self.path = path

    def __repr__(self) -> str:
        return f"{self.path} | {self.original_p:.2f} | - {self.optimized_lammps_p:.2f} | {self.optimized_sim_p:.2f}"

def calc_p_ratio(target_dir: str) -> float:
    with open(os.path.join(target_dir, "dump.lammpstrj")) as f:
        content = f.readlines()
        boxes = [index for index, line in enumerate(content) if line.startswith("ITEM: BOX BOUNDS")]
        
        initial_box_x = (float(content[boxes[0]+1].split(" ")[0]), float(content[boxes[0]+1].split(" ")[1]))
        initial_box_y = (float(content[boxes[0]+2].split(" ")[0]), float(content[boxes[0]+2].split(" ")[1]))
        initial_box = Box(
            initial_box_x[0], initial_box_x[1],
            initial_box_y[0], initial_box_y[1],
            -0.1, 0.1,
        )

        final_box_x = (float(content[boxes[-1]+1].split(" ")[0]), float(content[boxes[-1]+1].split(" ")[1]))
        final_box_y = (float(content[boxes[-1]+2].split(" ")[0]), float(content[boxes[-1]+2].split(" ")[1]))
        final_box = Box(
            final_box_x[0], final_box_x[1],
            final_box_y[0], final_box_y[1],
            -0.1, 0.1,
        )

        dy = final_box.y - initial_box.y
        dx = final_box.x - initial_box.x
        return -dy/dx

def do_work(n_atoms: int):
    """Creates a directory for a certain type of networks,
    distinguished by approx. number of atoms, number of atom types
    and compression value.

    Runs LJ simulations, creates a network out of resulting disk arrangement,
    applies a random perturbation to a graph then compresses it.

    Parameters
    ----------
    n_atoms : int

    batch_size : int

    """
    lj_sim = LJSimulation(
        n_atoms=n_atoms,
        n_atom_types=4,
        atom_sizes=atom_sizes[0:4],
        box_dim=box_dim,
        temperature_range=temperature_range,
        n_steps=N_STEPS,
    )
    
    comp_sim = CompressionSimulation(
        network_filename="network.lmp",  # do not change!
        strain=0.02,  # % of box X dimension
        strain_rate=1e-5,  # speed of compression
        temperature_range=TemperatureRange(1e-7, 1e-7, 1.0),
        dump_frequency=1000,  # `None` if you want 2000 steps or put a value to dump every N steps
    )

    custom_dir = os.path.join(calculation_directory, f"{n_atoms}_{4}")
    os.makedirs(custom_dir)
    
    assert(os.path.exists(custom_dir) and os.path.isdir(custom_dir))
    data_dir = os.path.join(custom_dir, "network_data")
    print(f"Data dir: {data_dir}")

    # Create a separate directory for each network
    for b in range(BATCH_SIZE):
        os.makedirs(os.path.join(data_dir, str(b + 1)))
    dirs = os.listdir(data_dir)
    dirs.sort(key=lambda x: int(x))

    # Work with each network one by one
    results: list[TestResult] = []
    for network_dir in dirs:
        print(f"Network dir: {network_dir}")
        target_dir = os.path.join(data_dir, network_dir)
        print(f"Target dir: {target_dir}")

        lj_sim.write_to_file(target_dir)
        run_lammps_calc(target_dir, input_file="lammps.in", mode="single")
        
        # Carefull with beads mass, too low and everything breaks
        new_network = Network.from_atoms(
            os.path.join(target_dir, "coord.dat"),
            include_angles=False,
            include_dihedrals=False,
            include_default_masses=100000.0,
            periodic=True,
        )

        new_network.write_to_file("network.lmp")

        # Write lammps input file and run compression simulation
        comp_sim._recalc_dump_freq(new_network.box.x)  
        comp_sim.write_to_file(target_dir)
        run_lammps_calc(
            target_dir,
            input_file="in.deformation",
            mode="single",
            num_threads=1,
            num_procs=1,
        )

        original_trajectory = parse_dump(os.path.join(target_dir, "dump.lammpstrj"), new_network, node_features="coord", skip=5)



        # run lammps again
        comp_sim.write_to_file(target_dir)
        run_lammps_calc(
            target_dir,
            input_file="in.deformation",
            mode="single",
            num_threads=1,
            num_procs=1,
        )

        # calculate optimized -dy/dx with lammps
        optimized_lammps_p = calc_p_ratio(target_dir)

        # simulated rollout
        inputs = ModelInputs(original_trajectory[0].cuda(), original_trajectory[1].cuda(), None)
        rollout = optimizer.simulate(inputs, 50)
        dy = calculate_delta(rollout[-1], rollout[0], 1, "cpu")
        dx = calculate_delta(rollout[-1], rollout[0], 0, "cpu")
        optimized_sim_p = -dy/dx

        result = TestResult(original_p, optimized_lammps_p, optimized_sim_p, target_dir)
        results.append(result)
    return results


with Pool() as p:
    results = p.map(do_work, n_atoms)

results = flatten(results)
with open(os.path.join(calculation_directory, "results.txt"), 'w') as f:
    for r in results:
        f.write(f"{r[0]} - {str(r[-1])}\n")