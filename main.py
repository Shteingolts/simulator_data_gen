from __future__ import annotations

import dataclasses
import os
import pickle
import shutil
import subprocess
import time
from copy import deepcopy
from multiprocessing import Pool

import numpy as np

from lammps_scripts import CompressionSimulation, ElasticScript, TemperatureRange, LJSimulation
from network import Bond, Network


def check_lammps_installation() -> str | None:
    """Checks wheter lammps is installed on the current machine

    Returns
    -------
    str | None
        Either path to the lammps executable or None if lammps is not found
    """
    try:
        out = subprocess.check_output(["which", "lmp"]).decode("utf-8")
        print(f"Lammps found: {out}")
        return out
    except subprocess.CalledProcessError:
        print("Lammps is probably not installed on your machine")
        return None


def clean_up(
    calculation_directory: str, network_filename: str = "original_network.lmp"
):
    """
    A helper function which deletes all the angle and dihedral data from the file.
    """
    network_filepath = os.path.join(calculation_directory, network_filename)
    network = Network.from_data_file(
        network_filepath, include_angles=True, include_dihedrals=False
    )
    network.write_to_file(network_filepath)


def get_elastic_data(log_file: str) -> ElasticData:
    with open(log_file, encoding="utf8") as file:
        content = file.readlines()
        p_ratio = float(content[-2].strip().split(" ")[3])
        shear_modulus = float(content[-4].strip().split(" ")[4])
        bulk_modulus = float(content[-8].strip().split(" ")[3])
        return ElasticData(p_ratio, bulk_modulus, shear_modulus)


def get_elastic_data_from_file(
    network_directory: str, network_file: str = "original_network.lmp"
) -> ElasticData:
    original_network_file = os.path.realpath(os.path.join(network_directory, network_file))
    network_file = os.path.realpath(os.path.join(network_directory, "network.lmp"))
    run_lammps(network_directory, mode="single", num_threads=1, num_procs=1)
    return get_elastic_data(os.path.join(network_directory, "log.lammps"))


def run_lammps(
    calculation_directory: str,
    input_file: str = "in.elastic",
    mode: str = "single",
    num_threads: int = 1,
    num_procs: int = 1,
):
    """
    A helper function which runs the external lammps code.
    """
    input_file_path = os.path.join(calculation_directory, input_file)
    mpi_command = f"mpirun -np {num_procs} lmp -in {input_file}".split()
    command = f"lmp -in {input_file_path}".split()

    os.chdir(os.path.dirname(input_file_path))
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    if mode == "single":
        subprocess.run(command, stdout=subprocess.DEVNULL)
    elif mode == "mpi":
        subprocess.run(mpi_command, stdout=subprocess.DEVNULL)


@dataclasses.dataclass
class ElasticData:
    """Combines the results of lammps calculation."""

    p_ratio: float
    bulk_modulus: float
    shear_modulus: float

    def __repr__(self) -> str:
        return f"Elastic data: P: {round(self.p_ratio, 3)}, B = {round(self.bulk_modulus, 3)}, G = {round(self.shear_modulus, 3)}"


@dataclasses.dataclass
class CalculationResult:
    """Dataclass which combines the result of the network optimization step."""

    bond: Bond
    elastic_data: ElasticData
    dG: float
    others: list[ElasticData] | None = None

    def __repr__(self) -> str:
        return f"Calc result: Bond: {self.bond}, {self.elastic_data}"


@dataclasses.dataclass
class CalculationSetup:
    """
    A dataclass accepted by `run_iteration()` function.
    A collection of arguments basically.
    """

    network_directory: str
    bonds_list: list[Bond]
    network_file: str = "original_network.lmp"


class StepResult:
    step_number: int
    elastic_data: ElasticData
    bond_removed: Bond
    network: Network
    closest_contenders: None | list[CalculationResult]

    def __init__(
        self,
        step_number: int,
        p_ratio: float,
        bond_removed: Bond,
        network: Network,
        closest_contenders: None | list[CalculationResult] = None,
    ) -> None:
        self.step_number = step_number
        self.network = network
        self.dG = p_ratio
        self.bond_removed = bond_removed
        self.closest_contenders = closest_contenders

    def __repr__(self) -> str:
        return (
            f"Step {self.step_number}: FF={self.dG}, Bond removed: {self.bond_removed}"
        )


def run_iteration(setup: CalculationSetup) -> CalculationResult:
    """
    Deletes every bond from the network once, then runs lammps compression simulation
    to calculate the Poisson's ratio. Returns the best bond to delete and all the others that had a
    positive impact, i.e., decreased the P ratio.
    Needs to have a file named `original_network.lmp` in the provided directory.

    Parameters
    ----------
    network_directory : str
        Directory containing a network file and simulation files (see example folder)
    network_file : str, optional
        filename for the network file, by default "original_network.lmp"

    Returns
    -------
    CalculationResult

    """

    original_network_file = os.path.join(setup.network_directory, setup.network_file)
    log_file = os.path.join(setup.network_directory, "log.lammps")

    original_network = Network.from_data_file(
        original_network_file,
        include_angles=True,
        include_dihedrals=False)
    
    original_network.write_to_file(os.path.join(setup.network_directory, "network.lmp"))
    initial_elastic_data: ElasticData = get_elastic_data_from_file(setup.network_directory)
    current_elastic_data = initial_elastic_data
    best_dG: float = 10000000000000.0
    bond_to_be_deleted: Bond | None = None
    others: list[Bond] = []
    start_local = time.perf_counter()
    for index, bond in enumerate(setup.bonds_list):
        
        # deepcopy because we don't want to modify the original network
        current_network = deepcopy(original_network)
        
        # remove a bond, recalculate angles
        current_network.remove_bond(bond)
        reduced_angles = current_network._compute_angles(
            current_network.atoms, current_network.box, 0.01
        )
        current_network.angles = reduced_angles
        current_network.header.angles = len(reduced_angles)
        current_network.header.angle_types = len(reduced_angles)
        
        # write a network to a file
        current_network.write_to_file("network.lmp")

        # run lammps on the previously written network
        run_lammps(setup.network_directory, mode="single")

        # get the new elastic data for the network with N-1 bonds
        new_elastic_data = get_elastic_data(log_file)
        dG = abs(initial_elastic_data.shear_modulus - new_elastic_data.shear_modulus)

        # nominate the bond for deletion if the change in G
        # is lower than the one previously recorder
        if dG < best_dG and len(bond.atom1.bonded) >= 4 and len(bond.atom2.bonded) >= 4:
            best_dG = dG
            current_elastic_data = new_elastic_data
            bond_to_be_deleted = bond
        # if dG is not lower than the best, but still lower than initial,
        # add it to the others pool
        elif (
            dG >= best_dG
            and len(bond.atom1.bonded) >= 4
            and len(bond.atom2.bonded) >= 4
        ):
            others.append(CalculationResult(bond, new_elastic_data, dG, None))
        else:
            pass
            # don't need to do anything when the removal
            # of the bond results in the increase of P ratio
    
    end_local = time.perf_counter()
    print(f"{len(setup.bonds_list)} checked in {end_local-start_local:.3f} seconds")
    return CalculationResult(bond_to_be_deleted, current_elastic_data, best_dG, others)


def load_optimization_log(pickle_file: str) -> list[StepResult]:
    optimization_log: list[StepResult] = []
    with open(pickle_file, "rb") as log_file:
        try:
            while True:
                optimization_log.append(pickle.load(log_file))
        except EOFError:
            pass
    return optimization_log


def parallel(
    main_calculation_directory: str,
    n_procs: int = 6,
    dG_threshold: float = 0.01
) -> dict[int, StepResult]:
    """
    The main function which performs the network optimization.

    For every iterative step of the optimization, a new directory named "step_N" is created,
    where N is the serial number for the current iteration, starting from 1.
    Multiprocessing is implemented in way such that for every core requested a seperate directory
    named "core_N" is created within the "step_N" directory.
    The total number of bond deletions are separated into M batches, where M=n_bonds/_ncores. Each core
    is assigned its own batch. After all indivial iteration are complete, one bond is removed from the network which resulted
    in the highest decrease of fitness function.

    The process runs until either:
    - 1. There is no change in fitness function observed between iterations, or
    - 2. Fitness function reaches a certain threshold difined by `ff_threshold` variable.

    The number of core available for the process is controled by the `n_procs` variable.
    """
    
    # create a log file. Overwrite if it's there already
    opt_log_file_path = os.path.join(main_calculation_directory, "optimization_log.pkl")
    opt_log_file = open(opt_log_file_path, "wb")
    opt_log_file.close()
    
    intermidiate_results = {}
    step_counter = 1
    start_time = time.perf_counter()
    while True:
        # create a directory for the current calculation step
        current_working_directory = os.path.realpath(
            os.path.join(main_calculation_directory, f"step_{step_counter}")
        )
        try:
            os.makedirs(current_working_directory)
        except OSError:
            print(f"Previous directory {current_working_directory} exists and is going to be replaced.")
            shutil.rmtree(current_working_directory)
            os.makedirs(current_working_directory)

        # if it's not the first iteration, copy `result.lmp` file
        # from the previous step directory and rename it accordingly
        print(f"Step {step_counter}")
        if step_counter != 1:
            shutil.copy(
                os.path.join(main_calculation_directory, f"step_{step_counter-1}", "result.lmp"),
                os.path.join(current_working_directory),
            )
            os.rename(
                os.path.join(current_working_directory, "result.lmp"),
                os.path.join(current_working_directory, "original_network.lmp"),
            )

        if step_counter == 1:
            network = Network.from_data_file(
                os.path.join(main_calculation_directory, "original_network.lmp"),
                include_angles=True,
                include_dihedrals=False,
            )
        else:
            network = Network.from_data_file(
                os.path.join(current_working_directory, "original_network.lmp")
            )

        # divide all the bonds in the network into a number of tasks equal to the number of cores
        n_tasks = len(network.bonds) // n_procs
        remainder = len(network.bonds) % n_procs
        tasks = []
        start = 0
        for i in range(n_procs):
            size = n_tasks + (i < remainder)
            tasks.append(network.bonds[start : start + size])
            start += size

        # for each task, create a separate directory within the step directory
        # copy the calculation files
        # create a list of calculation setups
        calculations: list[CalculationSetup] = []
        for i, task in enumerate(tasks):
            core_dir = os.path.realpath(os.path.join(current_working_directory, f"core_{i+1}"))
            os.makedirs(core_dir)
            calculations.append(CalculationSetup(core_dir, task))
            elastic_calc = ElasticScript("network.lmp")
            elastic_calc.write_to_file(core_dir)

        # copy the network file into a new directory
        if step_counter == 1:
            for i in range(len(tasks)):
                shutil.copy(
                    os.path.join(main_calculation_directory, "original_network.lmp"),
                    os.path.join(current_working_directory, f"core_{i+1}"),
                )
                shutil.copy(
                    os.path.join(main_calculation_directory, "original_network.lmp"),
                    os.path.join(current_working_directory),
                )
        else:
            # also copy the same file into every directory
            for i in range(len(tasks)):
                shutil.copy(
                    os.path.join(
                        main_calculation_directory,
                        f"step_{step_counter-1}",
                        "result.lmp",
                    ),
                    os.path.join(current_working_directory, f"core_{i+1}"),
                )
                os.rename(
                    os.path.join(
                        current_working_directory, f"core_{i+1}", "result.lmp"
                    ),
                    os.path.join(
                        current_working_directory, f"core_{i+1}", "original_network.lmp"
                    ),
                )

        with Pool(n_procs) as pool:
            results: list[CalculationResult] = pool.map(run_iteration, calculations)

        # filter out results where the deletion of a bond did not improve the fitness function
        results = [result for result in results if result.bond is not None]

        if results:
            # sort list of CalculationResults by the value of fitness function from lowest to highest
            results = sorted(results, key=lambda result: result.dG)
            best_result = results[0]
            others: list[CalculationResult] = []
            for index, result in enumerate(results):
                if index == 0 and len(result.others) > 0:
                    for other in result.others:
                        others.append(other)
                else:
                    others.append(CalculationResult(result.bond, result.elastic_data, result.dG))
                    for other in result.others:
                        others.append(other)

            others = sorted(others, key=lambda result: result.dG)
            print(f"Step {step_counter} => P={format(best_result.elastic_data.p_ratio, '.4f')}")

            okay_to_continue = True
            if not okay_to_continue:
                # raise NotImplementedError('do not use that')
                end_time = time.perf_counter()
                print(f"Threshold {dG_threshold} achieved at step {step_counter}.\nElapsed time: {format(start_time - end_time, '.3f')} s")
                network = Network.from_data_file(
                    os.path.join(current_working_directory, "original_network.lmp")
                )
                network.remove_bond(best_result.bond)
                reduced_angles = network._compute_angles(network.atoms, network.box, 0.01)
                network.angles = reduced_angles
                network.header.angles = len(reduced_angles)
                network.header.angle_types = len(reduced_angles)
                intermidiate_results[step_counter] = StepResult(
                    step_counter,
                    best_result.elastic_data,
                    best_result.bond,
                    network,
                    closest_contenders=others,
                )
                network.write_to_file(os.path.join(main_calculation_directory, "final_result.lmp"))
                print(f"The optimized network was written in {os.path.join(main_calculation_directory, "final_result.lmp")}")

                with open(opt_log_file_path, "ab") as opt_log_file:
                    pickle.dump(
                        StepResult(
                            step_counter,
                            best_result.elastic_data,
                            best_result.bond,
                            network,
                            closest_contenders=others,
                        ),
                        opt_log_file,
                    )

                # do the compression
                compression_directory = os.path.join(main_calculation_directory, 'compression') 
                os.makedirs(compression_directory, exist_ok=True)
                network.set_angle_coeff(0.01)
                network.write_to_file(os.path.join(compression_directory, 'final_result.lmp'))
                comp_sim = CompressionSimulation(
                    strain_direction='x',
                    box_size=network.box.x,
                    network_filename='final_result.lmp',
                    strain=0.03,
                    temperature_range=TemperatureRange(1e-7, 1e-7, 10)
                )
                comp_sim._recalc_dump_freq(network.box.x)
                comp_sim.write_to_file(compression_directory)
                run_lammps(compression_directory, input_file='in.deformation')

                # clean up space after
                print("Cleaning up calculation directories...")
                step_dirs = [obj for obj in os.listdir(main_calculation_directory) if obj.startswith("step_")]
                for d in step_dirs:
                    shutil.rmtree(os.path.join(main_calculation_directory, d))

                print("Done!")
                break
            if okay_to_continue:
                network = Network.from_data_file(os.path.join(current_working_directory, "original_network.lmp"))
                network.remove_bond(best_result.bond)
                reduced_angles = network._compute_angles(network.atoms, network.box, 0.01)
                network.angles = reduced_angles
                network.header.angles = len(reduced_angles)
                network.header.angle_types = len(reduced_angles)
                intermidiate_results[step_counter] = StepResult(
                    step_counter,
                    best_result.elastic_data,
                    best_result.bond,
                    network,
                    closest_contenders=others,
                )
                network.write_to_file(os.path.join(current_working_directory, "result.lmp"))
                with open(opt_log_file_path, "ab") as opt_log_file:
                    pickle.dump(
                        StepResult(
                            step_counter,
                            best_result.elastic_data,
                            best_result.bond,
                            network,
                            closest_contenders=others,
                        ),
                        opt_log_file,
                    )
                step_counter += 1
        else:
            # if no improvement was observed at all, end the process
            end_time = time.perf_counter()
            print(f"No change in fitness function at step {step_counter}.\nWall time: {format(start_time - end_time, '.3f')} s")
            final_network_file = os.path.join(main_calculation_directory, "final_result.lmp")
            network.write_to_file(final_network_file)
            print(f"The optimized network was written in {final_network_file}")

            # do the compression
            compression_directory = os.path.join(main_calculation_directory, 'compression') 
            print(f"Network compression calculation: {compression_directory}")
            os.makedirs(compression_directory, exist_ok=True)
            network.set_angle_coeff(0.01)
            network.write_to_file(os.path.join(compression_directory, 'final_result.lmp'))
            comp_sim = CompressionSimulation(
                strain_direction='x',
                box_size=network.box.x,
                network_filename='final_result.lmp',
                strain=0.03,
                temperature_range=TemperatureRange(1e-7, 1e-7, 10)
            )
            comp_sim._recalc_dump_freq(network.box.x)
            comp_sim.write_to_file(compression_directory)
            run_lammps(compression_directory, input_file='in.deformation')

            # clean up space after
            print("Cleaning up calculation directories...")
            step_dirs = [obj for obj in os.listdir(main_calculation_directory) if obj.startswith("step_")]
            for d in step_dirs:
                shutil.rmtree(os.path.join(main_calculation_directory, d))

            print("Done!")
            break

    return intermidiate_results


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    wd = os.path.realpath("123")
    os.makedirs(wd)
    
    n_atoms = np.linspace(100, 100, 1, dtype=int)
    batch_size = 1
    # make networks first
    for size in n_atoms:
        for n in range(batch_size):
            cwd = os.path.join(wd, str(size), str(n))
            os.makedirs(cwd)

            # LJ sim
            lj_dir = os.path.join(cwd, "LJ_sim")
            os.makedirs(lj_dir)
            lj_sim = LJSimulation(
                size,
                n_atom_types=4,
                temperature_range=TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0),
                n_steps=30000
            )
            lj_sim.write_to_file(lj_dir)
            run_lammps(lj_dir, input_file="lammps.in", mode="single")
                
            # Create a network from LJ coords. 
            new_network = Network.from_atoms(
                os.path.join(lj_dir, "coord.dat"),
                periodic=True,
                include_default_masses=1e6, # arbitrary mass for interesting compression
                include_angles=True,
                include_dihedrals=False
            )
            # set angle coeffs to 0.01
            new_network.set_angle_coeff(0.01)

            # save network
            new_network.write_to_file(os.path.join(cwd, "original_network.lmp"))

            # write files for elastic script
            elasticsim = ElasticScript("original_network.lmp")
            elasticsim.write_to_file(cwd)

            # call optimization function
            parallel(
                cwd,
                10,
                1e-16)



