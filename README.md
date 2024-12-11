# GNN Simulator Data Generator (OUTDATED)

## Overview
The `GNN Simulator Data Generator`, as the name suggests, automates the generation of random disordered elastic networks and the simulation of their compression using LAMMPS.

## Requirements
- Python 3.10 or higher (no third-party libraries required)
- LAMMPS software installed on the user's system with lmp alias added

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/Shteingolts/simulator_data_gen
2. Navigate to the cloned directory:
    ```sh
    cd simulator_data_generator
3. Run the script `simulation.py`:
    ```sh
    python3 simulation.py
    ```

## Usage
Modify various simulation parameters from inside the `gen_sim_data()` function inside `simulation.py`.
