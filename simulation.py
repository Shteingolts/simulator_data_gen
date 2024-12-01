"""
OLD FILE DON'T USE
"""

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


def construct_network(data_dir: str, outfile_name: str, beads_mass: float = 1e6):
    new_network = network.Network.from_atoms(
        os.path.join(data_dir, "coord.dat"),
        include_angles=False,
        include_dihedrals=False,
        include_default_masses=beads_mass,
        periodic=True,
    )
    new_network.write_to_file(outfile_name)
    return new_network


def gen_sim_data(
    custom_dir: str = "",
    lj_sim: LJSimulation = LJSimulation(),
    comp_sim: CompressionSimulation = CompressionSimulation(),
    n_networks: int = 5,
):
    """Change simulation parameters (`LJSimulation` and `CompressionSimulation` class attributes)
    to control how networks are generated and compressed.
    Note: do not expect any combination of parameters to work!

    Parameters
    ----------
    custom_dir : str, optional
        write data into a custom directory, by default ""
    lj_sim: LJSimulation
        configuration for LJ simulation, working defaults
    comp_sim: CompressionSimulation
        configuration for compression simulation, working defaults
    n_networks : int, optional
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
    for n in range(n_networks):
        os.makedirs(os.path.join(data_dir, str(n + 1)))
    dirs = os.listdir(data_dir)
    dirs.sort(key=lambda x: int(x))

    # Work with each network one by one
    for n, network_dir in enumerate(dirs):
        target_dir = os.path.join(data_dir, network_dir)
        t_start = perf_counter()

        lj_sim.write_to_file(target_dir)
        run_lammps_calc(target_dir, input_file="lammps.in", mode="single")
        
        # carefull with beads mass, too low and everything breaks
        new_network = construct_network(target_dir, "network.lmp", beads_mass=100000.0)

        comp_sim._recalc_dump_freq(new_network.box.x)

        comp_sim.write_to_file(target_dir)
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
    lj_sim = LJSimulation(
        n_atoms=200,
        n_atom_types=4,
        atom_sizes=[1.6, 1.4, 1.2, 1.0],
        box_dim=[-7.0, 7.0, -7.0, 7.0, -0.1, 0.1],
        temperature_range=TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0),
        n_steps=30000,
    )
    comp_sim = CompressionSimulation(
        strain_direction='x',
        network_filename="network.lmp",  # do not change!
        strain=0.030,  # % of box X dimension
        strain_rate=1e-5,  # speed of compression
        desired_step_size=0.001,
        temperature_range=TemperatureRange(1e-7, 1e-7, 10.0),
    )

    gen_sim_data(custom_dir="", lj_sim=lj_sim, comp_sim=comp_sim, n_networks=60)
