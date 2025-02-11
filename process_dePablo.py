import os
import pickle
import subprocess
from multiprocessing.pool import Pool

import lammps_scripts


def run_lammps_calc(
    calculation_directory: str,
    input_file: str = "lammps.in",
    mode: str = "single",
    num_threads: int = 1,
    num_procs: int = 6,
):
    """
    A helper function which runs the external lammps code.
    """
    input_file_path = os.path.join(calculation_directory, input_file)
    mpi_command = f"mpirun -np {num_procs} lmp -in {input_file_path}".split()
    command = f"lmp -in {input_file_path}".split()

    os.chdir(calculation_directory)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if mode == "single":
        subprocess.run(command, stdout=subprocess.DEVNULL)
    elif mode == "mpi":
        subprocess.run(mpi_command, stdout=subprocess.DEVNULL)

# 
def process_network(path: str, step_size: float = 0.001):
    steps_to_check = (1, 15, 30, 49)
    with open(os.path.join(path, "optimization_log.pkl"), "rb") as f:
        opt_hist = pickle.load(f)
    for step in steps_to_check:
        comp_dir = os.path.join(path, f"comp_{step}_step_size_{step_size}")
        print(comp_dir)
        os.makedirs(comp_dir)
        opt_hist["network"][step].set_angle_coeff(0.00)
        for bond in opt_hist["network"][step].bonds:
            bond.bond_coefficient = bond.bond_coefficient * 10
        opt_hist["network"][step].write_to_file(os.path.join(comp_dir, "network.lmp"))
        comp_sim = lammps_scripts.CompressionSimulation(
            strain=0.01,
            strain_direction="x",
            box_size=opt_hist["network"][step].box.x,
            desired_step_size=step_size,
            temperature_range=lammps_scripts.TemperatureRange(1e-7, 1e-7, 10),
        )
        comp_sim.write_to_file(comp_dir)
        run_lammps_calc(comp_dir, input_file="in.deformation")


if __name__ == "__main__":
    path = "/home/sergey/work/simulator_data_gen/data/raw/200_networks"
    paths = [os.path.join(path, subdir) for subdir in os.listdir(path)][:40]
    print(f"Networks to process: {len(paths)}")
    # default=0.001, try 0.001/2, 0.001/4, 0.001/10,
    step_sizes = [0.002 for subdir in os.listdir(path)][:40]
    inputs = list(zip(paths, step_sizes))
    with Pool() as pool:
        pool.starmap(process_network, inputs)
