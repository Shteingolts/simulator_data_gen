from copy import deepcopy
import os
import random

import torch
from convert import assemble_data, network_from_data
from lammps_scripts import ElasticScript, CompressionSimulation, LJSimulation, TemperatureRange
from main import get_elastic_data
import network
from simulation import run_lammps_calc
from utils import get_correct_edge_attr

calc_dir = '/home/sergey/work/simulator_data_gen/randomly_pruned_dataset'
os.makedirs(calc_dir)

n_networks: int = 3
FRACTION_NODES: float = 0.3
FRACTION_EDGES: float = 0.3
MAX_STEPS: int = 200

def delete_dangling(atoms: list[network.Atom]) -> tuple[list, int]:
    new_atoms = [atom for atom in atoms if atom.n_bonds > 1]
    difference = len(atoms) - len(new_atoms)
    for atom in new_atoms:
        # erase information about the number of bonds and bonded neighbour ids
        atom.n_bonds = 0
        atom.bonded = []
    return (new_atoms, difference)


for i in range(n_networks):
    local_path = os.path.join(calc_dir, str(i))
    
    # create a network from scratch
    lj_sim = LJSimulation(
        n_atoms=random.randint(100, 250),
        box_dim=[-20, 20, -20, 20, -0.1, 0.1],
        temperature_range=TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0),
        n_steps=30000
    )
    lj_sim.write_to_file(local_path)
    run_lammps_calc(local_path, input_file='lammps.in')
    local_network = network.Network.from_atoms(os.path.join(local_path, 'coord.dat'), include_default_masses=1e6)
    local_network.set_angle_coeff(0.0)
    for bond in local_network.bonds:
        bond.bond_coefficient = 1/bond.length
    local_network.write_to_file(os.path.join(local_path, 'network.lmp'))

    networks: dict[float, network.Network] = {}
    buckets: dict[float, network.Network] = {
        -0.6 : None,
        -0.5 : None,
        -0.4 : None,
        -0.3 : None,
        -0.2 : None,
        -0.1 : None,
        +0.0 : None,
        +0.1 : None,
        +0.2 : None,
        +0.3 : None,
        +0.4 : None,
        +0.5 : None,
        +0.6 : None,
    }
    step_count = 0
    done = False
    while not done or step_count < MAX_STEPS:
        # perturb the network and collect 
        try_dir = os.path.join(local_path, 'perturb')
        os.makedirs(try_dir, exist_ok=True)
        
        new_net = deepcopy(local_network)
        diameters = [atom.diameter for atom in new_net.atoms]
        graph = assemble_data(new_net.atoms, new_net.bonds, box=new_net.box)

        # shift beads     
        nodes_to_shift = [random.randint(0, graph.x.shape[0]-1) for i in range(int(FRACTION_NODES*graph.x.shape[0]))]
        graph.x[nodes_to_shift] += torch.rand_like(graph.x[nodes_to_shift]) * 0.3
        graph.edge_attr = get_correct_edge_attr(graph)
        new_net = network_from_data(graph, box=graph.box)
        for atom, d in zip(new_net.atoms, diameters):
            atom.diameter = d
        new_net.masses = {1:1e6}
        for bond in new_net.bonds:
            bond.bond_coefficient = 1/bond.length

        # remove bonds
        bonds_to_remove = random.sample(new_net.bonds, int(len(new_net.bonds)*FRACTION_EDGES))
        new_net.bonds = list(filter(lambda b: b not in bonds_to_remove, new_net.bonds))
        new_net.header.bond_types = len(new_net.bonds)
        new_net.header.bonds = len(new_net.bonds)

        # remove dangling beads
        dangling_beads: int = 1
        while dangling_beads > 0:
            atoms, dangling_beads = delete_dangling(new_net.atoms)
            bonds = network.make_bonds(atoms, new_net.box, periodic=True)
            new_net.atoms = atoms
            new_net.bonds = bonds
            new_net.header = network.Header(atoms, bonds, new_net.box)
        for bond in new_net.bonds:
            bond.bond_coefficient = 1/bond.length

        # save the network and check the P ratio
        new_net.write_to_file(os.path.join(try_dir, 'network.lmp'))
        elastic_sim = ElasticScript(network_filename='network.lmp')
        elastic_sim.write_to_file(try_dir)
        run_lammps_calc(try_dir, input_file='in.elastic')
        elastic_data = get_elastic_data(log_file=os.path.join(try_dir, 'log.lammps'))
        step_count += 1
        if -0.6 > elastic_data.p_ratio or elastic_data.p_ratio > 0.6:
            continue
        else:
            round_p_ratio = round(elastic_data.p_ratio, 1)
            if buckets[round_p_ratio] is None:
                buckets[round_p_ratio] = new_net
                print(buckets)
        if not any([v is None for v in buckets.values()]):
            print(buckets)
            print('Finished everything')
            done=True
    
    print('Networks are done. Starting compression...\n')
    for index, net in enumerate(buckets.values()):
        comp_path = os.path.join(local_path, str(index))
        os.makedirs(comp_path, exist_ok=True)
        for bond in net.bonds:
            bond.bond_coefficient = 1/bond.length
        net.write_to_file(os.path.join(comp_path, 'network.lmp'))

        comp_sim = CompressionSimulation(
            box_size=net.box.x,
            temperature_range=TemperatureRange(1e-7, 1e-7, 10)
        )
        comp_sim.write_to_file(comp_path)
        run_lammps_calc(comp_path, input_file='in.deformation')
