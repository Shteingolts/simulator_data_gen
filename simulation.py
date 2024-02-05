import os
import subprocess
from time import perf_counter

import network
from lammps_scripts import CompressionSimulation, LJSimulation, TemperatureRange


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


def construct_network(data_dir: str, outfile_name: str, beads_mass: float = 1000000.0):
    new_network = network.Network.from_atoms(
        os.path.join(data_dir, "coord.dat"),
        include_angles=False,
        include_dihedrals=False,
        include_default_masses=beads_mass,
        periodic=True,
    )
    new_network.write_to_file(outfile_name)


def gen_sim_data(custom_dir: str = "", networks: int = 5):
    """Change simulation parameters (`LJSimulation` and `CompressionSimulation` class attributes)
    to control how networks are generated and compressed.
    Note: not all parameters you out in will work, experiment.

    Parameters
    ----------
    custom_dir : str, optional
        write data into a custom directory, by default ""
    networks : int, optional
        number of networks to make, by default 5
    """
    # checks if user-provided directory is valid
    # if nothing is provided, script directory is used
    if len(custom_dir) == 0:
        calc_dir = os.path.dirname(os.path.realpath(__file__))
    elif os.path.exists(custom_dir) and os.path.isdir(custom_dir):
        calc_dir = custom_dir
    else:
        print("Calculation directory is invalid. Using default directory")
        calc_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(calc_dir, "network_data")

    # Create a separate directory for each network
    for n in range(networks):
        os.makedirs(os.path.join(data_dir, str(n + 1)))
    dirs = os.listdir(data_dir)
    dirs.sort(key=lambda x: int(x))

    # Work with each network one by one
    for n, network_dir in enumerate(dirs):
        target_dir = os.path.join(data_dir, network_dir)

        # Run lammps calc to get coordinates and costruct a network
        lj_simulation = LJSimulation(
            n_atoms=300,
            n_atom_types=6,
            atom_sizes=[2, 1.8, 1.6, 1.4, 1.2, 1.0],
            box_dim=[-10.0, 10.0, -10.0, 10.0, -0.1, 0.1],
            temperature_range=TemperatureRange(T_start=0.001, T_end=0.001, bias=10.0),
            n_steps=50000
            )
        lj_simulation.write_to_file(target_dir)
        t_start = perf_counter()
        run_lammps_calc(target_dir, input_file="lammps.in", mode="single")
        construct_network(target_dir, "network.lmp", beads_mass=1000000.0) # carefull with beads mass, too low and everything breaks

        # Create deformation simulation and run it
        example_compression = CompressionSimulation(
            network_filename="network.lmp", # don't change
            strain=0.02, # % of box X dimension
            strain_rate=1e-5, # speed of compression
            temperature_range=TemperatureRange(0.01, 0.01, 10.0),
            dump_frequency=None # None if you want 2000 steps or put a value to dump every N steps
            )
        example_compression.write_to_file(target_dir)
        run_lammps_calc(
            target_dir,
            input_file="in.deformation",
            mode="mpi",
            num_threads=2,
            num_procs=2,
        )
        t_stop = perf_counter()
        print(f"{target_dir} - {round(t_stop-t_start, 2)} s.")


if __name__ == "__main__":
    gen_sim_data(networks=5)
