"""
A small collection of functions which parse data from lammps dump file
into a PyTorch Geometric `Data` object and back.

Compared to the previous data parcing pipeline, introduces two extra attribiutes
into the PyTroch Data object, namely `box` and `atom_ids`, which are needed for
the lammps trajectory file format.

Intended as a substite for the helpers.py file in the future.
"""
from array import ArrayType
import os

import numpy as np
import torch
from torch_geometric.data import Data

import network
from network import Atom, Bond, Header, Box, Network
import utils


def assemble_data(
    atoms: list[network.Atom],
    bonds: list[network.Bond],
    box: network.Box,
    node_features: str = "coord",
) -> Data:
    """A helper function which assembles the PyTorch_Geometric `Data` object.
    
    Parameters
    ----------
    `atoms` : list
        list of `network.Atom` objects transformed into nodes
    `bonds` : list
        list of `network.Bond` objects transformed into edges
    `node_features` : str, optional
        which node features to include, by default "coord"

    Returns
    -------
    Data
        PyTorch_Geometric `Data` object
    """
    # mapping of atomic IDs to their list indices
    atom_ids = torch.tensor([atom.atom_id for atom in atoms])
    id_to_index_map = {atom.atom_id: i for atom, i in zip(atoms, range(len(atoms)))}

    # edges as defined with lammps IDs
    edges_with_ids = [(bond.atom1.atom_id, bond.atom2.atom_id) for bond in bonds]

    # edges as defined with indices
    edges_with_indices = [
        (id_to_index_map[node1], id_to_index_map[node2])
        for node1, node2 in edges_with_ids
    ]

    edge_index = torch.tensor(np.array(edges_with_indices).T)

    #NOTE: edge vectors will always be defined as vectors between the nodes in the SAME simulation box.
    # They will be wrond for bonds that cross the simulation box boundary.
    # On the other hand, edge lengths will always be correct.
    edge_vectors = [torch.tensor([bond.atom1.x - bond.atom2.x, bond.atom1.y - bond.atom2.y]) for bond in bonds] 
    edge_vectors = torch.stack(edge_vectors)
    edge_lengths = torch.tensor([bond.length for bond in bonds])
    edge_coeffs = torch.tensor([bond.bond_coefficient for bond in bonds])

    # edge vector and its length
    edge_attr = [
        torch.cat((v, torch.tensor([length]), torch.tensor([stiffness]))) 
        for v, length, stiffness in zip(edge_vectors, edge_lengths, edge_coeffs)
    ]
    edge_attr = torch.stack(edge_attr)

    match node_features:
        case "dummy":
            node_features = torch.tensor([[0] for i in range(len(atoms))])
        case "vel":
            try:
                node_features = torch.tensor([[atom.vx, atom.vy] for atom in atoms])
            except AttributeError:
                raise Exception("Atoms don't have velocity info")
        case "coord":
            node_features = torch.tensor([[atom.x, atom.y] for atom in atoms])
        case "full":
            node_features = torch.tensor(
                [[atom.x, atom.y, atom.vx, atom.vy] for atom in atoms]
            )
        case _:
            raise Exception(f"{node_features} not recognized as feature!")

    tmp = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, box=box, atom_ids=atom_ids)
    correct_edge_vectors = utils.get_correct_edge_vec(tmp)
    updated_edge_attr = torch.column_stack([correct_edge_vectors, edge_lengths, edge_coeffs])

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=updated_edge_attr,
        box=box,
    )

def compute_bonds_fast(input_network: network.Network, nodes: ArrayType | None, edge_index: ArrayType | None, box_size: tuple[float] | None):
    input_network.fix_sort()
    if nodes is None:
        nodes  = np.array([(atom.x, atom.y) for atom in input_network.atoms])
    if edge_index is None:
        edge_index = np.array([(bond.atom1.atom_id-1, bond.atom2.atom_id-1) for bond in input_network.bonds])
    if box_size is None:
        box_size_x = input_network.box.x
        box_size_y = input_network.box.y
    else:
        box_size_x = box_size[0]
        box_size_y = box_size[1]
    
    adj_list = {}
    for edge in edge_index:
        try:
            adj_list[edge[0]].append(edge[1])
        except KeyError:
            adj_list[edge[0]] = [edge[1]]
        try:
            adj_list[edge[1]].append(edge[0])
        except KeyError:
            adj_list[edge[1]] = [edge[0]]
    
    bonds = set()
    edge_attr = set()
    for center_node in range(len(nodes)):
        center_position = nodes[center_node]
        neighbours = adj_list[center_node]
        for n in neighbours:
            neighbour_position = nodes[n]
            vec = center_position - neighbour_position
            vec[0] = np.where(abs(vec[0]) >= box_size_x // 2,
                vec[0] - box_size_x*np.sign(vec[0]),
                vec[0]
            )
            
            vec[1] = np.where(abs(vec[1]) >= box_size_y // 2,
                vec[1] - box_size_y*np.sign(vec[1]),
                vec[1]
            )
            bonds.add(
                Bond(input_network.atoms[center_node], input_network.atoms[n], np.linalg.norm(vec), 1/np.power(np.linalg.norm(vec), 2))
            )
            edge_attr.add(torch.tensor([vec[0].item(), vec[1].item(), np.linalg.norm(vec).item(), 1/np.power(np.linalg.norm(vec), 2).item()]))

    return list(bonds), torch.stack(list(edge_attr))


def parse_dump(
    dump_filepath: str,
    original_network: network.Network,
    node_features: str = "full",
    skip: int = 1,
) -> list[Data]:
    """Parses a lammps dump file. By default returns x, y, vx, and vy as node features.

    Parameters
    ----------
    `dump_filepath` : str
        Path to the lammps trajectory files
    `original_network` : network.Network
        simulated network to get the accurate information about periodic bonds
    `node_features` : str, optional
        node features to include into Data object, by default "full".
        See `assemble_data()` function for more info.
    `skip` : int, optional
        load each n-th step starting from the first, by default 1 (skip none)

    Returns
    -------
    list[Data]
        list of pytorch_geometric Data objects
    """
    with open(dump_filepath, "r", encoding="utf8") as f:
        content = f.readlines()

    timesteps: list[int] = []
    for index, line in enumerate(content):
        if "ITEM: TIMESTEP" in line:
            timesteps.append(index)

    original_edge_index = [
        (bond.atom1.atom_id, bond.atom2.atom_id) for bond in original_network.bonds
    ]
    bond_map = {
        (bond.atom1.atom_id, bond.atom2.atom_id): bond
        for bond in original_network.bonds
    }

    data_list = []
    for i in range(0, len(timesteps) - 1, skip):
        timestep_data = content[timesteps[i] : timesteps[i + 1]]

        # get box info
        x1, x2 = (
            float(timestep_data[5].split(" ")[0]),
            float(timestep_data[5].split(" ")[1]),
        )
        y1, y2 = (
            float(timestep_data[6].split(" ")[0]),
            float(timestep_data[6].split(" ")[1]),
        )
        z1, z2 = (
            float(timestep_data[7].split(" ")[0]),
            float(timestep_data[7].split(" ")[1]),
        )
        box = network.Box(x1, x2, y1, y2, z1, z2)

        # get atoms info
        new_atoms = []
        for atom_data in timestep_data[9:]:
            atom_data = atom_data.split()
            atom_diameter = [
                atom
                for atom in original_network.atoms
                if atom.atom_id == int(atom_data[0])
            ][0].diameter
            atom = network.Atom(
                atom_id=int(atom_data[0]),
                atom_diameter=atom_diameter,
                x=float(atom_data[1]),
                y=float(atom_data[2]),
                z=float(atom_data[3]),
            )
            atom.vx = float(atom_data[4])
            atom.vy = float(atom_data[5])
            atom.vz = float(atom_data[6])
            new_atoms.append(atom)

        new_atom_map = {atom.atom_id: atom for atom in new_atoms}
        new_bonds = []
        for edge_index in original_edge_index:
            id1, id2 = edge_index[0], edge_index[1]
            new_atom1 = new_atom_map[id1]
            new_atom2 = new_atom_map[id2]
            bond_stiffness = bond_map[(id1, id2)].bond_coefficient

            # calculate the proper distance between two bonded atoms 
            # keeping periodicity in mind
            if abs(new_atom1.x - new_atom2.x) > box.x / 2:
                real_x2 = max(new_atom1.x, new_atom2.x)
                real_x1 = min(new_atom1.x, new_atom2.x) + box.x
            else:
                real_x1 = new_atom1.x
                real_x2 = new_atom2.x
            if abs(new_atom1.y - new_atom2.y) > box.y / 2:
                real_y2 = max(new_atom1.y, new_atom2.y)
                real_y1 = min(new_atom1.y, new_atom2.y) + box.y
            else:
                real_y1 = new_atom1.y
                real_y2 = new_atom2.y

            new_length = ((real_x2 - real_x1) ** 2 + (real_y2 - real_y1) ** 2) ** 0.5
            new_bond = network.Bond(new_atom1, new_atom2)
            new_bond.length = new_length
            new_bond.bond_coefficient = bond_stiffness
            new_bonds.append(new_bond)

        data_list.append(assemble_data(new_atoms, new_bonds, box, node_features=node_features))

    return data_list


def bulk_load(
    data_dir: str, n_networks: int, node_features: str = "full", skip: int = 1
) -> list[list[Data]]:
    """Loads data from a provided directory. By default returns x, y, vx, and vy as node features.
    Assumes that each directory in the `data_dir` contains a network simulation.

    Parameters
    ----------
    data_dir : str
        Path to the data directory

    Returns
    -------
    list[list[Data]]
    """
    sim_dirs = [
        os.path.abspath(os.path.join(data_dir, directory))
        for directory in os.listdir(data_dir)
    ]
    data = []
    for index, sim_dir in enumerate(sim_dirs):
        # reading network from `coord.dat` instead of `*.lmp` to get the accurate information about periodic bonds
        current_network = network.Network.from_atoms(
            os.path.join(sim_dir, "coord.dat"),
            include_angles=False,
            include_dihedrals=False,
        )
        current_network.write_to_file(os.path.join(sim_dir, "true_network.lmp"))
        dump_file = os.path.join(sim_dir, "dump.lammpstrj")
        print(f"{index+1}/{len(sim_dirs)} : {dump_file}")
        data.append(parse_dump(dump_file, current_network, node_features=node_features, skip=skip))

        # stop loading of desired number of network simulation is parsed
        if index + 1 >= n_networks:
            break
    return data


def network_from_data(data_object: Data, box: Box | None = None) -> network.Network:
    """Transforms PyTorch Data object into Network object.

    Parameters
    ----------
    `data_object` : Data
        PyTorch Data object containing the graph
    `template` : network.Network
        original network compressed with lammps from which data object was made

    Returns
    -------
    network.Network
        updated network
    """
    # create atoms 
    atoms = []
    for index, node in enumerate(data_object.x):
        atom_id = index + 1
        x = float(node[0])
        y = float(node[1])
        atoms.append(Atom(atom_id=atom_id, atom_diameter=0.0, x=x, y=y, z=0.0))
    
    # update bonds
    bonds=[]
    for index, ((source_id, target_id), (ux, uy, length, k)) in enumerate(zip(data_object.edge_index.T, data_object.edge_attr)):
        atom1 = atoms[int(source_id)]
        atom1.n_bonds += 1
        atom2 = atoms[int(target_id)]
        atom2.n_bonds += 1
        bond_coefficient = float(k)
        length = float(length)
        
        bond = Bond(atom1, atom2)
        bond.bond_coefficient = bond_coefficient
        bond.length = length
        
        bonds.append(bond)

    if box is None:
        box = data_object.box
    
    header = Header(
        atoms,
        bonds,
        box,
    )

    return Network(atoms, bonds, box, header, masses={1: 100000.0})


def dump(data: list[Data], filedir: str = "", filename: str = "dump_custom.lammpstrj"):
    """Writes a lammps trajctory file from the list of PyTorch Geometric
    Data objects.

    Parameters
    ----------
    `data` : list[Data]
        list of Data objects to dump into a file as trajectory
    `filedir` : str, optional
        directory to write the output file to, by default ""
    `filename` : str, optional
        output file name, by default "dump_custom.lammpstrj"
    """
    filepath = os.path.join(filedir, filename)
    print(filepath)
    with open(filepath, "w", encoding="utf8") as f:
        for index, data_object in enumerate(data):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{index}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{data_object.x.shape[0]}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{data_object.box.x1} {data_object.box.x2}\n")
            f.write(f"{data_object.box.y1} {data_object.box.y2}\n")
            f.write(f"{data_object.box.z1} {data_object.box.z2}\n")
            f.write(
                "ITEM: ATOMS id x y z vx vy vz\n"
            )  # TODO: make this atoms header dynamic
            for node_index, node in enumerate(data_object.x):
                atom_line = f"{data_object.atom_ids[node_index]} {node[0]} {node[1]} {0} {node[2]} {node[3]} {0}\n"
                f.write(atom_line)


def bulk_dump(data_dir: str):
    # TODO: write this function
    raise NotImplementedError


if __name__ == "__main__":
    pass