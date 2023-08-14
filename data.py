import os
import random
import subprocess
import shutil
from distutils.dir_util import copy_tree
import network
import helpers

NETWORKS: int = 50
DATAPOINTS: int = 2000
COMPRESSION_DEGREE = 0.05

data_dir = os.getcwd()
print(data_dir)

for n in range(NETWORKS):
    os.makedirs(os.path.join(data_dir, str(n+1)))

def run_lammps_calc(
    calculation_directory: str,
    input_file: str = 'lammps.in',
    mode: str = "single",
    num_threads: int = 1,
    num_procs: int = 1,
):
    """
    A helper function which runs the external lammps code.
    """
    input_file_path = os.path.join(calculation_directory, input_file)
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
        os.path.join(directory, 'coord.dat'),
        include_angles=False,
        include_dihedrals=False,
        include_default_masses=10000,
        )
    new_network.write_to_file(network_filename)

def extract_coords_to_csv(input_file: str):
    with open(input_file, 'r', encoding='utf8') as f:
        content = f.readlines()
        timesteps: dict[int, dict[int, tuple]] = {}
        for line in content:
            

dirs = os.listdir(data_dir)
dirs = [directory for directory in dirs if (os.path.isdir(directory) and directory.isnumeric())]
dirs.sort()

for n, network_dir in enumerate(dirs):

    target_dir = os.path.join(data_dir, network_dir)
    # print(f'Target dir {n}: {target_dir}')
    # put lammps calc file into the directory
    shutil.copy(os.path.abspath(os.path.join(data_dir, 'lammps.in')), os.path.abspath(target_dir))
    with open(os.path.join(target_dir, 'lammps.in'), 'r', encoding='utf8') as f:
        content = f.readlines()
        for n, line in enumerate(content):
            if 'create_atoms' in line:
                line = line.split()
                line[4] = str(random.randint(1, 999999))
                line = ' '.join(line) + '\n'
                content[n] = line
    
    with open(os.path.join(target_dir, 'lammps.in'), 'w', encoding='utf8') as f:
        f.writelines(content)

    # run lammps calc to get coordinates
    run_lammps_calc(target_dir, input_file='lammps.in')
    
    # construct_network
    construct_network(target_dir, 'network.lmp')

    # copy compression simulation files and a network file into a subdirectory
    compression_dir = os.path.abspath(os.path.join(target_dir, 'compression_sim'))
    original_compression_files = os.path.abspath(os.path.join(data_dir, 'compression_sim'))
    # print(f'Compression dir: {compression_dir}')
    copy_tree(original_compression_files, compression_dir)
    shutil.copy(
        os.path.abspath(os.path.join(target_dir, 'network.lmp')),
        compression_dir)

    # run compression simulation
    run_lammps_calc(compression_dir, input_file='in.deformation')
    

