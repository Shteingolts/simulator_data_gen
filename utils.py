from copy import deepcopy
from itertools import chain
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import Tensor
from matplotlib.patches import Patch
from torch_geometric.data import Data

from lammps_scripts import LJSimulation, TemperatureRange
from network import Box
import network


def flatten(iterable: list) -> list:
    return list(chain.from_iterable(iterable))


def draw_graph(
    graph: Data | list[Data],
    edges: bool = True,
    periodic_edges: bool = True,
    box: bool = False,
    node_color: str = "skyblue",
    standalone: bool = True,
):
    if standalone:
        plt.figure(figsize=(graph.box.x / 3, graph.box.y / 3))
    G = nx.Graph()
    # Add nodes
    for index, node in enumerate(graph.x):
        G.add_node(index)

    # Add edges
    if edges:
        edge_index = graph.edge_index.numpy().T
        for edge in edge_index:
            edge_len = torch.norm(graph.x[edge[1]] - graph.x[edge[0]])
            if not periodic_edges:
                if edge_len > (graph.box.x / 2 + graph.box.y / 2) / 2:
                    continue
                else:
                    G.add_edge(edge[0], edge[1])
            else:
                G.add_edge(edge[0], edge[1])

    pos = {index: (float(node[0]), float(node[1])) for index, node in enumerate(graph.x)}
    
    if box:
        B = nx.Graph()
        box_corners = [i for i in range(len(graph.x)+1, len(graph.x)+5)]
        box_edges = [
            [box_corners[0] ,box_corners[1]],
            [box_corners[1], box_corners[2]],
            [box_corners[2], box_corners[3]],
            [box_corners[3], box_corners[0]]]
        
        for corner, edge in zip(box_corners, box_edges):
            B.add_node(corner)
            B.add_edge(edge[0], edge[1])
        b_pos = {}
        b_pos[box_corners[0]] = (graph.box.x1, graph.box.y2)
        b_pos[box_corners[1]] = (graph.box.x2, graph.box.y2)
        b_pos[box_corners[2]] = (graph.box.x2, graph.box.y1)
        b_pos[box_corners[3]] = (graph.box.x1, graph.box.y1)

        nx.draw_networkx(B, b_pos, with_labels=False, node_color="black", node_size=1)

    nx.draw_networkx(G, pos, with_labels=False, node_color=node_color, node_size=20, font_size=10)
    if standalone:
        plt.show()


def visualize_graphs(
    original_graph: Data,
    updated_graph: Data,
    edges: bool = True,
    periodic_edges: bool = True,
    adjust: bool = False,
    node_colors: list[str] = ["blue", "red"],
):
    if adjust:
        x_average = torch.mean(original_graph.x[:, 0])
        y_average = torch.mean(original_graph.x[:, 1])
        center_mass1 = torch.Tensor([x_average, y_average])

        x_average = torch.mean(updated_graph.x[:, 0])
        y_average = torch.mean(updated_graph.x[:, 1])
        center_mass2 = torch.Tensor([x_average, y_average])

        shift = center_mass1 - center_mass2
        updated_graph.x +=shift
    
    draw_graph(
        original_graph,
        edges=edges,
        periodic_edges=periodic_edges,
        node_color=node_colors[0],
        standalone=False,
    )
    draw_graph(
        updated_graph,
        edges=edges,
        periodic_edges=periodic_edges,
        node_color=node_colors[1],
        standalone=False,
    )

    legend_handles = [
        Patch(facecolor=node_colors[0], edgecolor=node_colors[0], label="original"),
        Patch(facecolor=node_colors[1], edgecolor=node_colors[1], label="predicted"),
    ]
    plt.legend(handles=legend_handles)

    plt.show()


def filter_data(data: list[list[Data]], skip: int) -> list[list[Data]]:
    new_data = []
    for simulation in data:
        new_sim = []
        for index, graph in enumerate(simulation):
            if index % skip == 0:
                new_sim.append(graph)
        new_data.append(new_sim)
    return new_data


def deviation(array, mean=None) -> float:
    """
    Parameters
    ----------
    array : torch.Tensor
        specififcally 1D array
    mean : float or None, optional
        by default None

    Returns
    -------
    float
    """
    if mean is None:
        return torch.max(array-torch.mean(array))
    else:
        return torch.max(array)


def load_data(filepath: str, skip: int = 1) -> list[list[Data]]:
    with open(filepath, 'rb') as f:
        data = torch.load(f)
    
    return filter_data(data, skip=skip)


def add_pruned_bonds_back(original_network: network.Network, pruned_network: network.Network):
    pruned_bonds = list(set(original_network.bonds) - set(pruned_network.bonds))
    for b in pruned_bonds:
        b.bond_coefficient = 0.0000

    with_dummies = deepcopy(pruned_network)
    combined_bonds = with_dummies.bonds + pruned_bonds
    with_dummies.bonds = combined_bonds
    with_dummies.header.bonds = len(combined_bonds)
    with_dummies.header.bond_types = len(combined_bonds)
    return with_dummies


def get_correct_edge_vec(original_graph: Data) -> Tensor:
    # Extract source and target node coordinates
    original_source_nodes = original_graph.x[original_graph.edge_index[0]][:, 0:2]
    original_target_nodes = original_graph.x[original_graph.edge_index[1]][:, 0:2]

    # Extract box dimensions
    box = original_graph.box

    # Compute naive edge vectors
    naive_edge_vectors = original_source_nodes - original_target_nodes

    # Compute corrections for x-coordinates
    adjust_x = torch.abs(naive_edge_vectors[:, 0]) > (box.x / 2)
    correction_x = torch.sign(original_source_nodes[:, 0]) * (
        box.x / 2
        - torch.abs(original_source_nodes[:, 0])
        + box.x / 2
        - torch.abs(original_target_nodes[:, 0])
    )
    fixed_target_nodes_x = torch.where(
        adjust_x,
        original_source_nodes[:, 0] + correction_x,
        original_target_nodes[:, 0],
    )

    # Compute corrections for y-coordinates
    adjust_y = torch.abs(naive_edge_vectors[:, 1]) > (box.y / 2)
    correction_y = torch.sign(original_source_nodes[:, 1]) * (
        box.y / 2
        - torch.abs(original_source_nodes[:, 1])
        + box.y / 2
        - torch.abs(original_target_nodes[:, 1])
    )
    fixed_target_nodes_y = torch.where(
        adjust_y,
        original_source_nodes[:, 1] + correction_y,
        original_target_nodes[:, 1],
    )

    # Combine the corrected coordinates
    fixed_target_nodes = torch.column_stack(
        [fixed_target_nodes_x, fixed_target_nodes_y]
    )

    # Compute the fixed edge vectors
    fixed_edge_vectors = fixed_target_nodes - original_source_nodes
    return fixed_edge_vectors


def get_correct_edge_attr(original_graph: Data) -> Tensor:
    edge_vectors = get_correct_edge_vec(original_graph)
    lengths = torch.norm(edge_vectors, dim=1)
    bond_coeffs = 1 / torch.pow(lengths, 2)
    return torch.column_stack((edge_vectors, lengths, bond_coeffs))


def get_correct_distances(graph: Data, updated_graph: Data) -> Tensor:
    original_source_nodes = graph.x[graph.edge_index[0]]
    original_target_nodes = graph.x[graph.edge_index[1]]

    updated_source_nodes = updated_graph.x[graph.edge_index[0]]
    updated_target_nodes = updated_graph.x[graph.edge_index[1]]
    
    source_displacements = updated_source_nodes - original_source_nodes
    target_displacements = updated_target_nodes - original_target_nodes

    original_edge_vectors = get_correct_edge_vec(graph)[:, :2]

    updated_edge_vectors = (original_edge_vectors + target_displacements) - source_displacements
    updated_distances = torch.norm(updated_edge_vectors, dim=1)
    bond_coeffs = graph.edge_attr[:, -1]
    updated_edge_attr = torch.column_stack([updated_edge_vectors, updated_distances, bond_coeffs])
    return updated_edge_attr


def get_periodic_estimation(graph: Data, box: Box) -> Tensor:
    source_nodes = graph.x[graph.edge_index[0]]
    target_nodes = graph.x[graph.edge_index[1]]
    # edge_vectors = source_nodes - target_nodes
    
    box_x = box.x
    box_y = box.y

    # distances in x and y directions
    dx = target_nodes[:, 0] - source_nodes[:, 0]
    dy = target_nodes[:, 1] - source_nodes[:, 1]

    # if any if the distances are larger than half of the box,
    # it means that a wrong periodic copy of the node was used.
    original_target_nodes_x = torch.where(dx > box_x/2, target_nodes[:, 0] - box_x, target_nodes[:, 0])
    original_target_nodes_x = torch.where(dx < -box_x/2, original_target_nodes_x + box_x, original_target_nodes_x)

    original_target_nodes_y = torch.where(dy > box_y/2, target_nodes[:, 1] - box_y, target_nodes[:, 1])
    original_target_nodes_y = torch.where(dy < -box_y/2, original_target_nodes_y + box_y, original_target_nodes_y)

    target_nodes = torch.column_stack((original_target_nodes_x, original_target_nodes_y))
    
    edge_vectors = target_nodes - source_nodes

    lengths = torch.norm(edge_vectors, dim=1)

    bond_coeffs = graph.edge_attr[:, -1]
    new_edge_attr = torch.column_stack([edge_vectors, lengths, bond_coeffs])
    return new_edge_attr


def draw_network(
    net: network.Network,
    edges: bool = True,
    periodic_edges: bool = True,
    box: bool = False,
    node_color: str = "skyblue",
    standalone: bool = True,
    node_size: float = 20,
    figure_scale: float = 3
):
    if standalone:
        plt.figure(figsize=(8, 8))
    G = nx.Graph()
    # Add nodes
    for atom in net.atoms:
        G.add_node(atom.atom_id)

    pos = {atom.atom_id: (float(atom.y), float(atom.x)) for atom in net.atoms}

    # Add edges
    if edges:
        edge_index = [(bond.atom1.atom_id, bond.atom2.atom_id) for bond in net.bonds]
        naive_edge_lengths = [bond.atom1.dist(bond.atom2) for bond in net.bonds]
        for edge, length in zip(edge_index, naive_edge_lengths):
            if periodic_edges:
                G.add_edge(edge[0], edge[1])
            else:
                if length < net.box.x // 2:
                    G.add_edge(edge[0], edge[1])

    if box:
        B = nx.Graph()
        box_corners = [(net.box.x1, net.box.y1), (net.box.x1, net.box.y2), (net.box.x2, net.box.y1), (net.box.x2, net.box.y2)]
        box_edges = [
            [box_corners[0] ,box_corners[1]],
            [box_corners[1], box_corners[2]],
            [box_corners[2], box_corners[3]],
            [box_corners[3], box_corners[0]]]
        
        for corner, edge in zip(box_corners, box_edges):
            B.add_node(corner)
            B.add_edge(edge[0], edge[1])
        b_pos = {}
        b_pos[box_corners[0]] = (net.box.x1, net.box.y2)
        b_pos[box_corners[1]] = (net.box.x2, net.box.y2)
        b_pos[box_corners[2]] = (net.box.x2, net.box.y1)
        b_pos[box_corners[3]] = (net.box.x1, net.box.y1)

        nx.draw_networkx(B, b_pos, with_labels=False, node_color="black", node_size=1)

    nx.draw_networkx(G, pos, with_labels=False, node_color=node_color, node_size=node_size, font_size=10)
    # nx.draw_networkx(G, with_labels=False, node_color=node_color, node_size=node_size, font_size=10)
    if standalone:
        plt.show()


def remove_periodic(net: network.Network):
    non_periodic_bonds = []
    for bond in net.bonds:
        if bond.atom1.dist(bond.atom2) < net.box.x // 2:
            non_periodic_bonds.append(bond)
    net.bonds = non_periodic_bonds
    net.angles = net._compute_angles(net.atoms, net.box)
    net.header.bonds = len(non_periodic_bonds)
    net.header.bond_types = len(non_periodic_bonds)
    net.header.angles = len(net.angles)
    return net


def estimate_box(original_graph: Data, updated_graph: Data) -> Box:
    original_minmax_x = torch.max(original_graph.x[:, 0]) - torch.min(original_graph.x[:, 0])
    updated_minmax_x = torch.max(updated_graph.x[:, 0]) - torch.min(updated_graph.x[:, 0])

    original_minmax_y = torch.max(original_graph.x[:, 1]) - torch.min(original_graph.x[:, 1])
    updated_minmax_y = torch.max(updated_graph.x[:, 1]) - torch.min(updated_graph.x[:, 1])

    new_box_x = original_graph.box.x * updated_minmax_x / original_minmax_x
    new_box_y = original_graph.box.y * updated_minmax_y / original_minmax_y

    return Box(-new_box_x.item() / 2, new_box_x.item() / 2, -new_box_y.item() / 2, new_box_y.item() / 2, -0.1, 0.1)


def recalc_bond(input_network: network.Network, bond: network.Bond, periodic: bool = True):
    """
    this function recalculates the edges from the updated node positions after the perturbation.
    """
    if periodic:
        source_node = np.array((bond.atom1.x, bond.atom1.y))
        target_node = np.array((bond.atom2.x, bond.atom2.y))

        box_x = input_network.box.x
        box_y = input_network.box.y

        dx = np.abs(source_node[0] - target_node[0])
        dy = np.abs(source_node[1] - target_node[1])

        real_x1 = np.where(dx > box_x / 2, np.minimum(source_node[0], target_node[0]) + box_x, source_node[0])
        real_x2 = np.where(dx > box_x / 2, np.maximum(source_node[0], target_node[0]), target_node[0])

        real_y1 = np.where(dy > box_y / 2, np.minimum(source_node[1], target_node[1]) + box_y, source_node[1])
        real_y2 = np.where(dy > box_y / 2, np.maximum(source_node[1], target_node[1]), target_node[1])

        length = np.sqrt((real_x2 - real_x1)**2 + (real_y2 - real_y1)**2)
        bond_coeff = 1/(length**2)
        bond.length = length
        bond.bond_coefficient = bond_coeff
        return bond
    else:
        source_node = bond.atom1
        target_node = bond.atom2

        bond_length = source_node.dist(target_node)
        bond_coeff = 1 / (bond_length**2)
        return network.Bond(bond.atom1, bond.atom2, bond_length, bond_coeff)


def recalc_bonds(input_network: network.Network, periodic: bool = True):
    """
    this function recalculates the edges from the updated node positions after the perturbation.
    """
    new_bonds = [recalc_bond(input_network, bond, periodic=periodic) for bond in input_network.bonds]
    input_network.bonds = new_bonds
    return new_bonds


def inject_noise(input_network: network.Network, std: float | None = None, angle_coeffs: float = 0.01) -> network.Network:
    if std is not None:
        for atom in input_network.atoms:
            atom.x += random.gauss(mu=0, sigma=std)
            atom.y += random.gauss(mu=0, sigma=std)
    else:
        average_bond_length = sum([bond.length for bond in input_network.bonds])/len(input_network.bonds)
        for atom in input_network.atoms:
            atom.x += random.gauss(mu=0, sigma=average_bond_length/5)
            atom.y += random.gauss(mu=0, sigma=average_bond_length/5)
    _new_bonds = recalc_bonds(input_network, periodic=True)
    input_network.angles = input_network._compute_angles_fast(angle_coeffs)
    return input_network


def radius_graph(graph: Data, r: float) -> Tensor:
    G = nx.Graph()
    nodes = [(idx, {"pos":node}) for idx, node in enumerate(graph.x)]
    G.add_nodes_from(nodes)
    edge_index = torch.tensor(nx.geometric_edges(G, r)).T
    return edge_index


def add_edges_LJ(graph: Data, r: float, edge_attr: str = "vecs") -> Data:
    edge_index = radius_graph(graph, r=r)
    tmp = Data(
        x=graph.x,
        edge_index=edge_index,
        box=graph.box
    )
    # will not create any periodic edges most likely
    edge_vecs = get_correct_edge_vec(tmp)

    return Data(x=tmp.x, edge_index=tmp.edge_index, edge_attr=edge_vecs, box=tmp.box)


def randomize_LJ(n_atoms: int):
    atom_types = random.randint(2, 2)
    # atom_sizes = [random.uniform(0.3, 4.0) for i in range(atom_types)]
    print(atom_types)
    # print(atom_sizes)
    lj_sim = LJSimulation(
        n_atoms=n_atoms,
        n_atom_types=atom_types,
        atom_sizes=[0.3, 1.0],
        box_dim=[-40, 40, -40, 40, -0.1, 0.1],
        temperature_range=TemperatureRange(T_start=0.005, T_end=0.001, bias=10.0),
        Kn=0.5,
        n_parts=[40, 60],
        n_steps=30000
    )
    return lj_sim