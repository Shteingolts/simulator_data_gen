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

 
def process_network(path: str, step_size: float = 0.001, strain_rate: float = 1e-5):
    steps_to_check = (1, 10, 20, 30, 49)
    # try:
    with open(os.path.join(path, "optimization_log.pkl"), "rb") as f:
        opt_hist = pickle.load(f)
    for step in steps_to_check:
        comp_dir = os.path.join(path, f"comp_{step}_step_size_{step_size}_OOL_SR={strain_rate}")
        print(comp_dir)
        os.makedirs(comp_dir)
        opt_hist["network"][step].set_angle_coeff(0.00)
        for bond in opt_hist["network"][step].bonds:
            bond.bond_coefficient = 1/bond.length
        opt_hist["network"][step].write_to_file(os.path.join(comp_dir, "network.lmp"))
        comp_sim = lammps_scripts.CompressionSimulation(
            strain_direction="x",
            box_size=opt_hist["network"][step].box.x,
            strain=0.03,
            strain_rate=strain_rate,
            desired_step_size=step_size,
            temperature_range=lammps_scripts.TemperatureRange(1e-3, 1e-3, 10),
        )
        comp_sim.write_to_file(comp_dir)
        run_lammps_calc(comp_dir, input_file="in.deformation")
    # except FileNotFoundError:
    #     print(f"{path} opt log doesn't exist!!!")


if __name__ == "__main__":
    path = "/home/sergey/work/simulator_data_gen/one_over_l"
    paths = []
    for size_dir in os.listdir(path):
        if not size_dir.startswith("data_generation") and int(size_dir.split('_')[0]) < 140:
            for subdir in os.listdir(os.path.join(path, size_dir)):
                if int(subdir) < 6:
                    local_path = os.path.join(path, size_dir, subdir)
                    paths.append(local_path)
    
    for sr in [1e-6, 1e-5, 1e-4, 1e-3]:
        print(f"Networks to process: {len(paths)}")
        step_sizes = [0.001 for path in paths] # default=0.001
        strain_rates = [sr for path in paths]  # default=1e-5
        inputs = list(zip(paths, step_sizes, strain_rates))
        with Pool() as pool:
            pool.starmap(process_network, inputs)
