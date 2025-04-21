import os
import pickle
import subprocess
from multiprocessing.pool import Pool
from typing import Sequence

from lammps_scripts import CompressionSimulation, TemperatureRange


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


def process_network(
    path: str,
    steps_to_check: Sequence[int] = (1, 10, 20, 30, 49),
    step_size: float = 0.001,
    strain_rate: float = 1e-5,
    compression_temp: float = 1e-7,
):
    """Opens the dePablo optimization log located in `path` and compresses selected networks.

    Parameters
    ----------
    path : str
        Location of `optimization_log.pkl`
    steps_to_check : Sequence[int], optional
        What networks from optimization process to compress, by default (1, 10, 20, 30, 49)
    step_size : float, optional
        Controls dumping frequency during lammps compression.
        The value refers to the change in simulation box's x axis in one step, by default 0.001
    strain_rate : float, optional
        LAMMPS compression strain rate, by default 1e-5
    compression_temp : float, optional
        LAMMPS compression temperature, by default 1e-7
    """
    # open dePablo optimization history
    with open(os.path.join(path, "optimization_log.pkl"), "rb") as f:
        opt_hist = pickle.load(f)

    for step in steps_to_check:
        comp_dir = os.path.join(
            path, f"comp_{step}_step_size_{step_size}_OOL_step_size_{step_size}"
        )
        print(comp_dir)
        os.makedirs(comp_dir)

        # set angle coefficients to zero
        opt_hist["network"][step].set_angle_coeff(0.00)
        # redefine bond coeffs to 1/l just in case
        for bond in opt_hist["network"][step].bonds:
            bond.bond_coefficient = 1 / bond.length
        # save the network
        opt_hist["network"][step].write_to_file(os.path.join(comp_dir, "network.lmp"))
        temp_range = TemperatureRange(
            T_start=compression_temp, T_end=compression_temp, bias=10
        )
        comp_sim = CompressionSimulation(
            strain_direction="x",
            box_size=opt_hist["network"][step].box.x,
            strain=0.03,
            strain_rate=strain_rate,
            desired_step_size=step_size,
            temperature_range=temp_range,
        )
        comp_sim.write_to_file(comp_dir)
        run_lammps_calc(comp_dir, input_file="in.deformation")


if __name__ == "__main__":
    path = "/home/sergey/work/simulator_data_gen/one_over_l"
    paths = []
    for size_dir in os.listdir(path):
        if (
            not size_dir.startswith("data_generation")  # skip log file
            and int(size_dir.split("_")[0]) < 140
        ):
            for subdir in os.listdir(os.path.join(path, size_dir)):
                if int(subdir) < 6:
                    local_path = os.path.join(path, size_dir, subdir)
                    paths.append(local_path)

    print(f"Networks to process: {len(paths)}")
    step_sizes = [1e-3 for path in paths]  # default=0.001
    strain_rates = [1e-5 for path in paths]  # default=1e-5
    inputs = list(zip(paths, step_sizes, strain_rates))
    with Pool() as pool:
        pool.starmap(process_network, inputs)
