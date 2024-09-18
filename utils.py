from copy import deepcopy
from itertools import chain

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor
from matplotlib.patches import Patch
from torch_geometric.data import Data

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
    original_source_nodes = original_graph.x[original_graph.edge_index[0]]
    original_target_nodes = original_graph.x[original_graph.edge_index[1]]
    
    box = original_graph.box
    naive_edge_vectors = original_graph.x[original_graph.edge_index[0]] - original_graph.x[original_graph.edge_index[1]]
    # periodic_edge_indices = torch.where(torch.norm(naive_edge_vectors, dim=1) > (box.x / 2 + box.y / 2) / 2)
    # periodic_edge_vectors = naive_edge_vectors[periodic_edge_indices]
    # nonperiodic_edge_indices = torch.where(torch.norm(naive_edge_vectors, dim=1) <= (box.x / 2 + box.y / 2) / 2)
    # nonperiodic_edge_vectors = naive_edge_vectors[nonperiodic_edge_indices]

    # periodic_node_pairs = original_graph.edge_index.T[periodic_edge_indices]
    fixed_target_nodes_x = torch.zeros_like(original_target_nodes[:, 0])
    for index, edge in enumerate(naive_edge_vectors[:, 0]):
        if torch.abs(edge) > box.x / 2:
            changed = original_source_nodes[:, 0][index] +\
                torch.sign(original_source_nodes[:, 0][index]) * (
                    box.x / 2 - torch.abs(original_source_nodes[:, 0][index])
                    + box.x / 2 - torch.abs(original_target_nodes[:, 0][index])
                    )
            # print(f"ACTUAL EDGE INDEX: {index}")
            # print(f"ACTUAL EDGE: {naive_edge_vectors[index]}")
            # print(f"TARGET NODE INDEX {original_graph.edge_index[1][index]}")
            # print(f"TARGET NODE: {original_graph.x[original_graph.edge_index[1][index]]}")
            # print(f"CHANGED: {changed}")
            # print(f"FROM TARGET LIST: {original_target_nodes[index]}")
            # print(type(original_graph.edge_index[1][index].item()))
            fixed_target_nodes_x[index] = changed
    
    for i, v in enumerate(fixed_target_nodes_x):
        if v == 0:
            fixed_target_nodes_x[i] = original_target_nodes[:, 0][i]


    fixed_target_nodes_y = torch.zeros_like(original_target_nodes[:, 1])
    for index, edge in enumerate(naive_edge_vectors[:, 1]):
        if torch.abs(edge) > box.y / 2:
            changed = original_source_nodes[:, 1][index] +\
                torch.sign(original_source_nodes[:, 1][index]) * (
                    box.y / 2 - torch.abs(original_source_nodes[:, 1][index])
                    + box.y / 2 - torch.abs(original_target_nodes[:, 1][index])
                    )
            # print(f"ACTUAL EDGE INDEX: {index}")
            # print(f"ACTUAL EDGE: {naive_edge_vectors[index]}")
            # print(f"TARGET NODE INDEX {original_graph.edge_index[1][index]}")
            # print(f"TARGET NODE: {original_graph.x[original_graph.edge_index[1][index]]}")
            # print(f"CHANGED: {changed}")
            # print(f"FROM TARGET LIST: {original_target_nodes[index]}")
            # print(type(original_graph.edge_index[1][index].item()))
            fixed_target_nodes_y[index] = changed
    
    for i, v in enumerate(fixed_target_nodes_y):
        if v == 0:
            fixed_target_nodes_y[i] = original_target_nodes[:, 1][i]

    fixed_target_nodes = torch.column_stack([fixed_target_nodes_x, fixed_target_nodes_y])
    fixed_edge_vectors = fixed_target_nodes - original_source_nodes
    return fixed_edge_vectors


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


def estimate_box(original_graph: Data, updated_graph: Data) -> Box:
    original_minmax_x = torch.max(original_graph.x[:, 0]) - torch.min(original_graph.x[:, 0])
    updated_minmax_x = torch.max(updated_graph.x[:, 0]) - torch.min(updated_graph.x[:, 0])

    original_minmax_y = torch.max(original_graph.x[:, 1]) - torch.min(original_graph.x[:, 1])
    updated_minmax_y = torch.max(updated_graph.x[:, 1]) - torch.min(updated_graph.x[:, 1])

    new_box_x = original_graph.box.x * updated_minmax_x / original_minmax_x
    new_box_y = original_graph.box.y * updated_minmax_y / original_minmax_y

    return Box(-new_box_x.item() / 2, new_box_x.item() / 2, -new_box_y.item() / 2, new_box_y.item() / 2, -0.1, 0.1)
