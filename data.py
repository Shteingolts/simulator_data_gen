import os
import random
import shutil
import subprocess
from distutils.dir_util import copy_tree
from os import makedirs, path

import network

# import helpers

NETWORKS: int = 50
DATAPOINTS: int = 2000
COMPRESSION_DEGREE = 0.05


def run_lammps_calc(
    calculation_directory: str,
    input_file: str = "lammps.in",
    mode: str = "single",
    num_threads: int = 1,
    num_procs: int = 1,
):
    """
    A helper function which runs the external lammps code.
    """
    input_file_path = path.join(calculation_directory, input_file)
    mpi_command = f"mpirun -np {num_procs} lmp -in in.stress".split()
    command = f"lmp -in {input_file_path}".split()

    os.chdir(calculation_directory)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if mode == "single":
        subprocess.run(command, stdout=subprocess.DEVNULL)
    elif mode == "mpi":
        subprocess.run(mpi_command, stdout=subprocess.DEVNULL)


def construct_network(directory: str, network_filename: str):
    new_network = network.Network.from_atoms(
        os.path.join(directory, "coord.dat"),
        include_angles=False,
        include_dihedrals=False,
        include_default_masses=10000,
    )
    new_network.write_to_file(network_filename)


def extract_coords_to_csv(input_file: str):
    """
    Produces a csv file, where the rows correspond to the timesteps
    of the simulation and the columns to the atomic coordinates
    """
    with open(input_file, "r", encoding="utf8") as f:
        print(f'Input file: {input_file}')
        content = f.readlines()
        timesteps: dict[int, dict[int, tuple]] = {}
        box_dimensions: dict[int, tuple] = {}
        timestep_counter = 1
        for index, line in enumerate(content):
            if "TIMESTEP" in line:
                box_x = sum(map(lambda x: abs(float(x)), content[index + 5].split(" ")))
                box_y = sum(map(lambda x: abs(float(x)), content[index + 6].split(" ")))
                box_dimensions[timestep_counter] = (box_x, box_y)
                n_atoms = int(content[index + 3])
                atoms_start = index + 9
                atoms_end = atoms_start + n_atoms
                atoms: dict[int, tuple] = {}
                for atom_line in content[atoms_start:atoms_end]:
                    data_items = atom_line.split(" ")
                    atom_id = int(data_items[0])
                    atom_x = float(data_items[2])
                    atom_y = float(data_items[3])
                    atom_z = float(data_items[4])
                    atoms[atom_id] = (atom_x, atom_y, atom_z)
                timesteps[timestep_counter] = dict(sorted(atoms.items()))
                timestep_counter += 1

    timesteps = dict(sorted(timesteps.items()))
    box_dimensions = dict(sorted(box_dimensions.items()))

    csv_path = path.join(path.dirname(input_file), "data.csv")
    # print(f'Out file: {csv_path}')
    with open(csv_path, "w", encoding="utf8") as f:
        atoms = [atom_id for atom_id in timesteps[1]]
        # print(atoms)
        fields = []
        for atom in atoms:
            fields.append(str(atom) + "_x")
            fields.append(str(atom) + "_y")
            fields.append(str(atom) + "_z")
        fields.append("box_x")
        fields.append("box_y")
        # print(fields)
        f.write(";".join(fields) + "\n")

        for index, data in timesteps.items():
            data_str = [str(coord) for atom in data.values() for coord in atom]
            data_line = ";".join(data_str)
            # print(data_line)
            data_line = (
                ";".join(
                    [
                        data_line,
                        str(box_dimensions[index][0]),
                        str(box_dimensions[index][1]),
                    ]
                )
                + "\n"
            )
            f.write(data_line)


script_dir = os.path.dirname(os.path.realpath(__file__))
print(f"Script: {script_dir}")
data_dir = path.join(script_dir, "network_data")
print(f"Data: {data_dir}")


for n in range(NETWORKS):
    makedirs(path.join(data_dir, str(n + 1)))

dirs = os.listdir(data_dir)
dirs.sort(key=lambda x: int(x))

for n, network_dir in enumerate(dirs):
    target_dir = path.join(data_dir, network_dir)
    print(target_dir)
    print(os.listdir(target_dir))

    # copy lammps calcultion files into each network directory
    shutil.copy(path.join(script_dir, "lammps.in"), path.abspath(target_dir))

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
    # after changing the lines, rewrite the input file
    with open(path.join(target_dir, "lammps.in"), "w", encoding="utf8") as f:
        f.writelines(content)

    # run lammps calc to get coordinates
    run_lammps_calc(target_dir, input_file="lammps.in")

    # construct_network
    construct_network(target_dir, "network.lmp")

    # copy lammps compression simulation files and a network file into a subdirectory
    # within the network directory called `compression_sim`
    compression_dir = os.path.abspath(os.path.join(target_dir, "compression_sim"))
    original_compression_files = os.path.abspath(
        os.path.join(script_dir, "compression_sim")
    )
    copy_tree(original_compression_files, compression_dir)
    shutil.copy(
        os.path.abspath(os.path.join(target_dir, "network.lmp")), compression_dir
    )

    # run compression simulation
    run_lammps_calc(compression_dir, input_file="in.deformation")

input_files = [
    path.abspath(
        path.join(data_dir, network_dir, "compression_sim", "deform_dump.lammpstrj")
    )
    for network_dir in sorted(os.listdir(data_dir), key=lambda x: int(x))
]

for input_file in input_files:
    extract_coords_to_csv(input_file)
