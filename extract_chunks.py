import multiprocessing
import os
from multiprocessing import Pool, Queue
from os import listdir
from os.path import join

import torch

import network
from convert import parse_dump


def extract_from_dump(path: str):
    print(path)
    current_network = network.Network.from_data_file(
        join(path, "network.lmp"),
        include_angles=True,
        include_dihedrals=False,
    )

    sim = parse_dump(
        join(path, "dump.lammpstrj"),
        current_network,
        node_features='coord',
        skip=1,
    )
    return sim

def writer(output_dir: str, output_queue: Queue):
    """
    Write processed data to individual files using torch.save.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    chunk_id = 0
    while True:
        data = output_queue.get()
        if data is None:  # Sentinel value to exit
            break
        # Save each chunk as a separate file
        file_path = os.path.join(output_dir, f"chunk_{chunk_id}.pt")
        torch.save(data, file_path)
        chunk_id += 1


def main():
    raw_data_path = "/home/sergey/work/simulator_data_gen/1000_networks"
    binary_data_path = "/home/sergey/work/simulator_data_gen/data/binary/5000_dePablo_networks_OOL"
    
    # dePablo networks parse
    paths = []
    for size_dir in listdir(raw_data_path):
        if size_dir != "data_generation.log":
            for net_dir in os.listdir(os.path.join(raw_data_path, size_dir)):
                for comp_dir in os.listdir(os.path.join(raw_data_path, size_dir, net_dir)):
                    # if comp_dir.startswith("comp_") and comp_dir.endswith('_step_size_0.003'):
                    if comp_dir.startswith("comp_"):
                        current_dir = join(raw_data_path, size_dir, net_dir, comp_dir)
                        paths.append(current_dir)

    # Normal data parse
    # paths = []
    # for t in listdir(raw_data_path):
    #     if t != "data_generation.log":
    #         for comp_dir in os.listdir(os.path.join(raw_data_path, t, "network_data")):
    #             current_dir = join(raw_data_path, t, "network_data", comp_dir)
    #             paths.append(current_dir)

    
    # num_workers = multiprocessing.cpu_count()
    num_workers = 8

    # Create a queue for inter-process communication
    output_queue = multiprocessing.Queue()

    # Start the writer process
    writer_process = multiprocessing.Process(target=writer, args=(binary_data_path, output_queue))
    writer_process.start()

    with Pool(num_workers) as pool:
        # Process chunks in parallel and collect results
        for result in pool.imap_unordered(extract_from_dump, paths):
            # Send processed chunks to the output queue
            output_queue.put(result)

    # Send a sentinel value to indicate the writer process can terminate
    output_queue.put(None)

    # Wait for the writer process to finish
    writer_process.join()

if __name__ == "__main__":
    main()
