import os
import random
import shutil
import subprocess
from os import path

import network


def run_lammps_calc(
    calculation_directory: str,
    input_file: str = "lammps.in",
    mode: str = "single",
    num_threads: int = 1,
    num_procs: int = 14,
):
    """
    A helper function which runs the external lammps code.
    """
    input_file_path = path.join(calculation_directory, input_file)
    mpi_command = f"mpirun -np {num_procs} lmp -in {input_file_path}".split()
    command = f"lmp -in {input_file_path}".split()

    os.chdir(calculation_directory)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if mode == "single":
        subprocess.run(command, stdout=subprocess.DEVNULL)
    elif mode == "mpi":
        subprocess.run(mpi_command, stdout=subprocess.DEVNULL)


def construct_network(data_dir: str, outfile_name: str):
    new_network = network.Network.from_atoms(
        path.join(data_dir, "coord.dat"),
        include_angles=False,
        include_dihedrals=False,
        include_default_masses=100000,
        periodic=True,
    )
    new_network.write_to_file(outfile_name)


def gen_sim_data(custom_dir: str = "", networks: int = 5):
    # checks if user-provided directory is valid
    # if nothing is provided, default directory (script location)
    # is used
    if len(custom_dir) == 0:
        calc_dir = path.dirname(path.realpath(__file__))
    elif path.exists(custom_dir) and path.isdir(custom_dir):
        calc_dir == custom_dir
    else:
        print("Calculation directory is invalid. Using default directory")
        calc_dir = path.dirname(path.realpath(__file__))
    data_dir = path.join(calc_dir, "network_data")

    # Create a separate directory for each network
    for n in range(networks):
        os.makedirs(path.join(data_dir, str(n + 1)))
    dirs = os.listdir(data_dir)
    dirs.sort(key=lambda x: int(x))
    # Work with each network one by one
    for n, network_dir in enumerate(dirs):
        target_dir = path.join(data_dir, network_dir)
        print(target_dir)
        print(os.listdir(target_dir))
        # copy lammps calcultion files into each network directory
        shutil.copy(path.join(calc_dir, "lammps.in"), path.abspath(target_dir))
        # add random seed to lammps particle placement routine
        with open(path.join(target_dir, "lammps.in"), "r", encoding="utf8") as f:
            content = f.readlines()
            for n, line in enumerate(content):
                # For each network to be different, we need to put a random seed
                # into the lammps input file in the `create atoms` command.
                if "create_atoms" in line:
                    line = line.split()
                    line[4] = str(random.randint(1, 999999))
                    line = " ".join(line) + "\n"
                    content[n] = line
        # save the new lammps input file
        with open(path.join(target_dir, "lammps.in"), "w", encoding="utf8") as f:
            f.writelines(content)

        # Run lammps calc to get coordinates and costruct a network
        run_lammps_calc(target_dir, input_file="lammps.in")
        construct_network(target_dir, "network.lmp")

        # Copy lammps compression simulation files and a network file
        # into a subdirectory `sim` in the network directory
        simulation_directory = path.abspath(path.join(target_dir, "sim"))
        simulation_input_files = path.abspath(path.join(calc_dir, "lammps_compression_files"))
        shutil.copytree(simulation_input_files, simulation_directory, dirs_exist_ok=True)
        shutil.copy(path.abspath(path.join(target_dir, "network.lmp")), simulation_directory)

        # run compression simulation
        run_lammps_calc(simulation_directory, input_file="in.deformation", mode="mpi")

    # input_files = [
    #     path.abspath(
    #         path.join(data_dir, network_dir, "sim", "deform_dump.lammpstrj")
    #     )
    #     for network_dir in sorted(os.listdir(data_dir), key=lambda x: int(x))
    # ]


if __name__ == "__main__":
    gen_sim_data()
