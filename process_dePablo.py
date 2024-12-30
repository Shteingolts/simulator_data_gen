import os
import pickle
import shutil
import subprocess

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


steps_to_check = (0, 9, 19, 29, 39, 49)
path = "/home/sergey/work/gnn/NN_Simulator/20_networks/"
for d in os.listdir(path):
    local_d = os.path.join(path, d)
    with open(os.path.join(local_d, "optimization_log.pkl"), "rb") as f:
        opt_hist = pickle.load(f)
        for step in steps_to_check:
            comp_dir = os.path.join(local_d, f"comp_{step}")
            print(comp_dir)
            os.makedirs(comp_dir)
            opt_hist["network"][step].set_angle_coeff(0.00)
            opt_hist["network"][step].write_to_file(
                os.path.join(comp_dir, "network.lmp")
            )
            comp_sim = lammps_scripts.CompressionSimulation(
                strain_direction="x",
                box_size=opt_hist["network"][step].box.x,
                strain=0.03,
                temperature_range=lammps_scripts.TemperatureRange(1e-7, 1e-7, 10),
            )
            comp_sim.write_to_file(comp_dir)
            run_lammps_calc(comp_dir, input_file="in.deformation")
