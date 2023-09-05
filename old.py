from os import path


def extract_coords_to_csv(input_file: str):
    """
    Produces a csv file, where the rows correspond to the timesteps
    of the simulation and the columns to the atomic coordinates
    """
    with open(input_file, "r", encoding="utf8") as f:
        print(f'Input file: {input_file}')
        content = f.readlines()
        timesteps: dict[int, dict[int, tuple]] = {}
        box_dimensions: dict[int, tuple] = {}
        timestep_counter = 1
        for index, line in enumerate(content):
            if "TIMESTEP" in line:
                box_x = sum(map(lambda x: abs(float(x)), content[index + 5].split(" ")))
                box_y = sum(map(lambda x: abs(float(x)), content[index + 6].split(" ")))
                box_dimensions[timestep_counter] = (box_x, box_y)
                n_atoms = int(content[index + 3])
                atoms_start = index + 9
                atoms_end = atoms_start + n_atoms
                atoms: dict[int, tuple] = {}
                for atom_line in content[atoms_start:atoms_end]:
                    data_items = atom_line.split(" ")
                    atom_id = int(data_items[0])
                    atom_x = float(data_items[2])
                    atom_y = float(data_items[3])
                    atom_z = float(data_items[4])
                    atoms[atom_id] = (atom_x, atom_y, atom_z)
                timesteps[timestep_counter] = dict(sorted(atoms.items()))
                timestep_counter += 1

    timesteps = dict(sorted(timesteps.items()))
    box_dimensions = dict(sorted(box_dimensions.items()))

    csv_path = path.join(path.dirname(input_file), "data.csv")
    # print(f'Out file: {csv_path}')
    with open(csv_path, "w", encoding="utf8") as f:
        atoms = [atom_id for atom_id in timesteps[1]]
        # print(atoms)
        fields = []
        for atom in atoms:
            fields.append(str(atom) + "_x")
            fields.append(str(atom) + "_y")
            fields.append(str(atom) + "_z")
        fields.append("box_x")
        fields.append("box_y")
        # print(fields)
        f.write(";".join(fields) + "\n")

        for index, data in timesteps.items():
            data_str = [str(coord) for atom in data.values() for coord in atom]
            data_line = ";".join(data_str)
            # print(data_line)
            data_line = (
                ";".join(
                    [
                        data_line,
                        str(box_dimensions[index][0]),
                        str(box_dimensions[index][1]),
                    ]
                )
                + "\n"
            )
            f.write(data_line)