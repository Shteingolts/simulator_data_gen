import numpy as np
import torch
from torch_geometric.data import Data

import network


def network_to_data(input_network: network.Network, node_features: str) -> Data:
    # mapping of atomic IDs to their list indices
    id_to_index_map = {
        atom.atom_id: i
        for atom, i in zip(input_network.atoms, range(len(input_network.atoms)))
    }

    # edges as defined with lammps IDs
    edges_with_ids = [
        (bond.atom1.atom_id, bond.atom2.atom_id) for bond in input_network.bonds
    ]

    # edges as defined with indices
    edges_with_indices = [
        (id_to_index_map[node1], id_to_index_map[node2])
        for node1, node2 in edges_with_ids
    ]

    # final edge_index
    edge_index = torch.tensor(np.array(edges_with_indices).T)

    match node_features:
        case "dummy":
            nodes = torch.tensor(
                [[0] for i in range(len(input_network.atoms))], dtype=torch.float32
            )
        case "vel":
            try:
                nodes = torch.tensor(
                    [[atom.vx, atom.vy] for atom in input_network.atoms]
                )
            except:
                raise Exception("Probably no velocities info")

        case "coord":
            nodes = torch.tensor([[atom.x, atom.y] for atom in input_network.atoms])
        case _:
            raise Exception(f"{node_features} not recognized as feature!")

    edge_attr = torch.tensor(
        [[bond.length for bond in input_network.bonds]], dtype=torch.float32
    )

    return Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)


def assemble_data(atoms: list, bonds: list, node_features: str) -> Data:
    # mapping of atomic IDs to their list indices
    id_to_index_map = {atom.atom_id: i for atom, i in zip(atoms, range(len(atoms)))}

    # edges as defined with lammps IDs
    edges_with_ids = [(bond.atom1.atom_id, bond.atom2.atom_id) for bond in bonds]

    # edges as defined with indices
    edges_with_indices = [
        (id_to_index_map[node1], id_to_index_map[node2])
        for node1, node2 in edges_with_ids
    ]

    edge_index = torch.tensor(np.array(edges_with_indices).T)
    edge_attr = torch.tensor([[bond.length for bond in bonds]], dtype=torch.float32)

    match node_features:
        case "dummy":
            nodes = torch.tensor([[0] for i in range(len(atoms))], dtype=torch.float32)
        case "vel":
            try:
                nodes = torch.tensor([[atom.vx, atom.vy] for atom in atoms])
            except AttributeError:
                raise Exception(f"Atoms don't have velocity info")
        case "coord":
            nodes = torch.tensor([[atom.x, atom.y] for atom in atoms])
        case "full":
            nodes = torch.tensor([[atom.x, atom.y, atom.vx, atom.vy] for atom in atoms])
        case _:
            raise Exception(f"{node_features} not recognized as feature!")

    return Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)


def parse_dump(
    dump_file: str, original_network: network.Network, node_features: str
) -> list[Data]:
    with open(dump_file, "r", encoding="utf8") as f:
        content = f.readlines()

    timesteps = []
    for index, line in enumerate(content):
        if "ITEM: TIMESTEP" in line:
            timesteps.append(index)

    data_list = []
    for index, step in enumerate(timesteps[:-1]):
        timestep_data = content[step : timesteps[index + 1]]
        atoms = []
        bonds = original_network.bonds
        for atom_data in timestep_data[9:]:
            atom_data = atom_data.split()
            atom = network.Atom(
                atom_id=int(atom_data[0]),
                diameter=1.0,
                x=float(atom_data[1]),
                y=float(atom_data[2]),
                z=float(atom_data[3]),
            )
            atom.vx = float(atom_data[4])
            atom.vy = float(atom_data[5])
            atom.vz = float(atom_data[6])

            atoms.append(atom)

        data_list.append(assemble_data(atoms, bonds, node_features="full"))
    return data_list


if __name__ == "__main__":
    example_network = network.Network.from_data_file(
        "network_data/1/compression_sim/network.lmp"
    )
    example = parse_dump(
        "network_data/1/compression_sim/dump.lammpstrj", example_network, "full"
    )

    print(example[0].x)
