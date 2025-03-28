"""
A simple utility script, which takes lammps dump output and turns it
into a lammps-readable file.
v. 0.1.6
"""

from __future__ import annotations

from collections import defaultdict
import os
import pickle
import sys
from copy import deepcopy
from math import acos, degrees, sqrt
from typing import TextIO

import numpy as np


def table_row(items: list, widths: list, indent: str = "right") -> str:
    """
    Creates a string with the certain number of spaces between words
    alligned to either right or left
    """
    line = []
    for item, width in zip(items, widths):
        line.append(add_spaces(str(item), width, indent))

    return "   ".join(line) + "\n"


def add_spaces(string: str, width: int, indent: str = "right") -> str:
    """
    If the string is longer than provided width,
    returns the original string without change.
    """
    spaces_to_add = (width - len(string)) * " "
    if width <= len(string):
        return string
    elif indent == "right":
        return spaces_to_add + string
    elif indent == "left":
        return string + spaces_to_add
    else:
        raise IOError(f" [ERROR] {indent} is not a valid indent type. Only 'right' and 'left' are supported.")


class Atom:
    atom_id: int
    diameter: float
    n_bonds: int
    bonded: list[int]
    x: float
    y: float
    z: float
    disk_type: int | None

    def __init__(
        self,
        atom_id: int,
        atom_diameter: float,
        x: float,
        y: float,
        z: float,
        disk_type: int | None = None,
        atom_type: int = 1,
    ):
        self.atom_id = int(atom_id)
        self.atom_type = atom_type
        self.diameter = atom_diameter
        self.n_bonds = 0
        self.bonded = []
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.disk_type = disk_type


    def __repr__(self) -> str:
        return f"Atom {self.atom_id} : {self.x}, {self.y}, {self.z})."

    def __eq__(self, other: Atom) -> bool:
        if (
            self.atom_id == other.atom_id
            and self.diameter == other.diameter
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        ):
            return True
        return False

    def __hash__(self) -> int:
        return hash((self.atom_id, self.x, self.y, self.z))

    def move(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Atom:
        self.x += x
        self.y += y
        self.z += z
        return self

    def translate(self, box: Box, direction: tuple = (0, 0, 0)) -> Atom:
        x_move = direction[0] * box.x
        y_move = direction[1] * box.y
        z_move = direction[2] * box.z
        return self.move(x_move, y_move, z_move)

    def dist(self, atom: Atom) -> float:
        return (
            (self.x - atom.x) ** 2 + (self.y - atom.y) ** 2 + (self.z - atom.z) ** 2
        ) ** 0.5

    def within_box(self, box: Box) -> bool:
        x_min, x_max = min(box.x1, box.x2), max(box.x1, box.x2)
        y_min, y_max = min(box.y1, box.y2), max(box.y1, box.y2)
        z_min, z_max = min(box.z1, box.z2), max(box.z1, box.z2)

        if (
            x_min <= self.x <= x_max
            and y_min <= self.y <= y_max
            and z_min <= self.z <= z_max
        ):
            return True

        return False

    def on_edge(self, box: Box, delta: float) -> bool:
        delta_x = delta_y = delta_z = delta
        if box.x1 == box.x2 == 0.0:
            delta_x = 0.0
        if box.y1 == box.y2 == 0.0:
            delta_y = 0.0
        if box.z1 == box.z2 == 0.0:
            delta_z = 0.0

        smaller_box = Box(
            box.x1 + delta_x,
            box.x2 - delta_x,
            box.y1 + delta_y,
            box.y2 - delta_y,
            box.z1 + delta_z,
            box.z2 - delta_z,
        )
        bigger_box = Box(
            box.x1 - delta_x,
            box.x2 + delta_x,
            box.y1 - delta_y,
            box.y2 + delta_y,
            box.z1 - delta_z,
            box.z2 + delta_z,
        )

        if self.within_box(bigger_box) and not self.within_box(smaller_box):
            return True
        else:
            return False


class Bond:
    """
    All bonds have the same elasticity. The bond coefficient is 1 / d^2,
    where d is the bond length.
    """

    atom1: Atom
    atom2: Atom
    length: float
    bond_coefficient: float

    def __init__(self, atom1: Atom, atom2: Atom, length: None | float = None, bond_coeff: None | float = None):
        """Due to the periodicity of the network, when making a bond between two atoms
        one needs to find the closest pair of two atoms, which may not be in the same
        simulation box."""
        self.atom1 = atom1
        self.atom2 = atom2
        self.length = atom1.dist(atom2) if length is None else length
        self.bond_coefficient = 1 / (self.length**2) if bond_coeff is None else bond_coeff

    def __repr__(self) -> str:
        return f"""Bond(atom1: {self.atom1},
         atom2: {self.atom2}, 
         d: {self.length}, 
         coeff: {self.bond_coefficient})"""

    def __eq__(self, other: Bond) -> bool:
        if {self.atom1, self.atom2} == {other.atom1, other.atom2}:
            if round(self.length, 6) == round(other.length, 6):
                return True
        return False

    def __hash__(self) -> int:
        if self.atom1.atom_id > self.atom2.atom_id:
            return hash((self.atom2.atom_id, self.atom1.atom_id))
        else:
            return hash((self.atom1.atom_id, self.atom2.atom_id))


class Angle:
    """
    Value in degrees.
    """

    angle_id: int
    atom1: Atom
    atom2: Atom
    atom3: Atom
    energy: float
    value: float

    def __init__(
        self,
        angle_id: int,
        atom1: Atom,
        atom2: Atom,
        atom3: Atom,
        box: Box,
        energy: float = 0.0,
        value: float | None = None,
    ):
        self.angle_id = angle_id
        self.atom1 = atom1
        self.atom2 = atom2
        self.atom3 = atom3
        self.energy = energy

        if value is None:
            # stupid algorithm, but works.
            # find all possible copies,
            first_atom_candidates = []
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    for z in (-1, 0, 1):
                        first_atom_candidates.append(
                            deepcopy(atom1).translate(box, (x, y, z))
                        )

            third_atom_candidates = []
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    for z in (-1, 0, 1):
                        third_atom_candidates.append(
                            deepcopy(atom3).translate(box, (x, y, z))
                        )

            def closest(origin_atom: Atom, candidates: list[Atom]):
                closest_atom = candidates[0]

                for atom in candidates[1:]:
                    if origin_atom.dist(atom) < origin_atom.dist(closest_atom):
                        closest_atom = atom
                return closest_atom

            atom1 = closest(atom2, first_atom_candidates)
            atom3 = closest(atom2, third_atom_candidates)

            v12 = [atom1.x - atom2.x, atom1.y - atom2.y, atom1.z - atom2.z]
            v23 = [atom3.x - atom2.x, atom3.y - atom2.y, atom3.z - atom2.z]

            dot_product = v12[0] * v23[0] + v12[1] * v23[1] + v12[2] * v23[2]

            mag_v12 = sqrt(v12[0] ** 2 + v12[1] ** 2 + v12[2] ** 2)
            mag_v23 = sqrt(v23[0] ** 2 + v23[1] ** 2 + v23[2] ** 2)

            cos_angle = dot_product / (mag_v12 * mag_v23)
            angle = degrees(acos(cos_angle))
            self.value = angle
        else:
            self.value = value

    def __eq__(self, other: Angle) -> bool:
        if self.value == other.value:
            return True
        return False

    def __hash__(self) -> int:
        if self.atom1.atom_id > self.atom3.atom_id:
            return hash((self.atom3, self.atom2, self.atom1))
        else:
            return hash((self.atom1, self.atom2, self.atom3))

    def __repr__(self) -> str:
        return (
            f"Angle {self.angle_id} : "
            f"{self.atom1.atom_id}-{self.atom2.atom_id}"
            f"-{self.atom3.atom_id} | {round(self.value, 2)} "
            f"({round(180 - self.value, 2)}) deg."
        )


class Dihedral:
    atom1: Atom
    atom2: Atom
    atom3: Atom
    atom4: Atom

    def __init__(self) -> None:
        raise NotImplementedError("not yet...")


class Header:
    atoms: int
    bonds: int
    angles: int
    dihedrals: int
    impropers: int
    atom_types: int
    bond_types: int
    angle_types: int
    dihedral_types: int
    improper_types: int
    box_dimensions: tuple

    def __init__(
        self,
        atoms: list[Atom],
        bonds: list[Bond],
        box: Box,
        angles: list = [],
        dihedrals: list = [],
        impropers: list = [],
        atom_types: int = 1,  # defaults to one
        bond_types: int = 0,
        angle_types: int = 0,
        dihedral_types: int = 0,
        improper_types: int = 0,
    ):
        self.box_dimensions = box.dimensions
        self.atoms = len(atoms)
        self.bonds = len(bonds) if bonds else 0
        self.angles = len(angles) if angles else 0
        self.dihedrals = len(dihedrals) if dihedrals else 0
        self.impropers = len(impropers) if impropers else 0
        self.atom_types = atom_types

        if not self.bonds:
            self.bond_types = 0
        else:
            if bond_types == 0:
                self.bond_types = len(bonds)
            else:
                self.bond_types = bond_types

        if not self.angles:
            self.angle_types = 0
        else:
            if angle_types == 0:
                self.angle_types = len(angles)
            else:
                self.angle_types = angle_types

        if not self.dihedrals:
            self.dihedral_types = 0
        else:
            if dihedral_types == 0:
                self.dihedral_types = len(dihedrals)
            else:
                self.dihedral_types = dihedral_types

        if not self.impropers:
            self.improper_types = 0
        else:
            if improper_types == 0:
                self.improper_types = len(impropers)
            else:
                self.improper_types = improper_types

    def write_header(self, file: TextIO, add_box_line: bool = True) -> None:
        file.write("LAMMPS data file.\n\n")
        file.write(add_spaces(f"{str(self.atoms)}", 7))
        file.write(" atoms\n")
        file.write(add_spaces(f"{str(self.bonds)}", 7))
        file.write(" bonds\n")
        file.write(add_spaces(f"{str(self.angles)}", 7))
        file.write(" angles\n")
        file.write(add_spaces(f"{str(self.dihedrals)}", 7))
        file.write(" dihedrals\n")
        file.write(add_spaces(f"{str(self.impropers)}", 7))
        file.write(" impropers\n")
        file.write(add_spaces(f"{str(self.atom_types)}", 7))
        file.write(" atom types\n")
        file.write(add_spaces(f"{str(self.bond_types)}", 7))
        file.write(" bond types\n")
        file.write(add_spaces(f"{str(self.angle_types)}", 7))
        file.write(" angle types\n")
        file.write(add_spaces(f"{str(self.dihedral_types)}", 7))
        file.write(" dihedral types\n")
        file.write(add_spaces(f"{str(self.improper_types)}", 7))
        file.write(" improper types\n")
        file.write(add_spaces(f"{format(round(self.box_dimensions[0], 6), '.6f')}", 11))
        file.write(add_spaces(f"{format(round(self.box_dimensions[1], 6), '.6f')}", 11))
        file.write(" xlo xhi\n")
        file.write(add_spaces(f"{format(round(self.box_dimensions[2], 6), '.6f')}", 11))
        file.write(add_spaces(f"{format(round(self.box_dimensions[3], 6), '.6f')}", 11))
        file.write(" ylo yhi\n")
        file.write(add_spaces(f"{format(round(self.box_dimensions[4], 6), '.6f')}", 11))
        file.write(add_spaces(f"{format(round(self.box_dimensions[5], 6), '.6f')}", 11))
        file.write(" zlo zhi\n")
        if add_box_line:
            file.write("0.0 0.0 0.0 xy xz yz\n")


class Box:
    x1: float
    x2: float
    y1: float
    y2: float
    z1: float
    z2: float

    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1, self.x2 = min(x1, x2), max(x1, x2)
        self.y1, self.y2 = min(y1, y2), max(y1, y2)
        self.z1, self.z2 = min(z1, z2), max(z1, z2)

    def __repr__(self) -> str:
        return (
            f"Box ({round(self.x1, 3)} : {round(self.x2, 3)}) "
            f"({round(self.y1, 3)} : {round(self.y2, 3)}) "
            f"({round(self.z1, 3)} : {round(self.z2, 3)})"
        )

    @classmethod
    def from_data_file(cls, file: str) -> Box:
        with open(file, encoding="utf8") as data_file:
            content = data_file.readlines()

        for index, line in enumerate(content):
            if "xlo" in line:
                x1 = float(content[index].split()[0])
                x2 = float(content[index].split()[1])
                y1 = float(content[index + 1].split()[0])
                y2 = float(content[index + 1].split()[1])
                z1 = float(content[index + 2].split()[0])
                z2 = float(content[index + 2].split()[1])
                break
        else:  # no break
            print("[ERROR] Could not find box info")
            x1 = x2 = y1 = y2 = z1 = z2 = 0

        return Box(x1, x2, y1, y2, z1, z2)

    def resize(self, delta: float) -> Box:
        self.x1 += delta / 2
        self.x2 -= delta / 2
        self.y1 += delta / 2
        self.y2 -= delta / 2
        self.z1 += delta / 2
        self.z2 -= delta / 2
        return self

    @property
    def x(self):
        return abs(self.x2 - self.x1)

    @property
    def y(self):
        return abs(self.y2 - self.y1)

    @property
    def z(self):
        return abs(self.z2 - self.z1)

    @property
    def dimensions(self):
        return (self.x1, self.x2, self.y1, self.y2, self.z1, self.z2)


class Network:
    atoms: list[Atom]
    bonds: list[Bond]
    angles: list[Angle] | None
    dihedrals: list[Dihedral] | None
    masses: dict[int, float] | None
    box: Box
    header: Header

    def __init__(
        self,
        atoms: list[Atom],
        bonds: list[Bond],
        box: Box,
        header: Header,
        angles: list[Angle] | None = None,
        dihedrals: list[Dihedral] | None = None,
        masses: dict[int, float] | None = None,
    ):
        self.atoms = atoms
        self.bonds = bonds
        self.box = box
        self.header = header
        self.angles = angles
        self.dihedrals = dihedrals
        self.masses = masses


    def __repr__(self) -> str:
        return f"Network ({len(self.atoms)} atoms, {len(self.bonds)} bonds)."


    @property
    def coordination_number(self) -> float:
        return sum([len(atom.bonded) for atom in self.atoms]) / len(self.atoms)


    @property
    def atom_sizes(self) -> dict[float, int]:
        """Counts the number of atoms of different sizes in the network.

        Returns
        -------
        dict[float, int]
            Dictionary with atom sizes as keys and 
        """
        sizes = {}
        for atom in self.atoms:
            if atom.diameter not in sizes:
                sizes[atom.diameter] = 1
            else:
                sizes[atom.diameter] += 1
        return sizes


    @property
    def source_beads(self) -> tuple[int, ...]:
        return tuple([atom.atom_id for atom in self.atoms if atom.atom_type == 2])

    @property
    def target_beads(self) -> tuple[int, ...]:
        return tuple([atom.atom_id for atom in self.atoms if atom.atom_type == 3])

    @staticmethod
    def _compute_angles(atoms: list[Atom], box: Box) -> list[Angle]:
        atoms_map = {atom.atom_id: atom for atom in atoms}
        angles = set()
        for atom in atoms:
            for neighbour_k in atom.bonded:
                for neighbour_j in atom.bonded:
                    if atoms_map[neighbour_k] != atoms_map[neighbour_j]:
                        angles.add(
                            Angle(
                                len(angles) + 1,
                                atoms_map[neighbour_k],
                                atom,
                                atoms_map[neighbour_j],
                                box,
                            )
                        )
        return list(angles)


    def _compute_angles_fast(self, ang_coeff: float) -> list[Angle]:
        def compute_angle(vec1, vec2):
            return np.degrees(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
        
        self.fix_sort()
        self.atoms = sorted(self.atoms, key=lambda x: x.atom_id)

        nodes  = np.array([(atom.x, atom.y) for atom in self.atoms])
        edges = np.array([(bond.atom1.atom_id-1, bond.atom2.atom_id-1) for bond in self.bonds])



        adj_list = {}
        for edge in edges:
            try:
                adj_list[edge[0]].append(edge[1])
            except KeyError:
                adj_list[edge[0]] = [edge[1]]
            try:
                adj_list[edge[1]].append(edge[0])
            except KeyError:
                adj_list[edge[1]] = [edge[0]]

        angles = []
        for center_node in adj_list:
            center_position = nodes[center_node]
            neighbour_nodes = adj_list[center_node]
            vecs = []
            for n in neighbour_nodes:
                neighbour_position = nodes[n]
                vec = center_position - neighbour_position
                vec[0] = np.where(abs(vec[0]) >= self.box.x // 2,
                    vec[0] - self.box.x*np.sign(vec[0]),
                    vec[0]
                )
                
                vec[1] = np.where(abs(vec[1]) >= self.box.y // 2,
                    vec[1] - self.box.y*np.sign(vec[1]),
                    vec[1]
                )
                vecs.append(vec)

            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    node1, node3 = neighbour_nodes[i], neighbour_nodes[j]
                    value = compute_angle(vecs[i], vecs[j])
                    angles.append(
                        Angle(
                            0,
                            self.atoms[node1],
                            self.atoms[center_node],
                            self.atoms[node3],
                            self.box,
                            ang_coeff,
                            value,
                        )
                    )

        for i, angle in enumerate(angles):
            angle.angle_id = i+1
        
        return angles


    def _compute_dihedrals(self):
        raise NotImplementedError("not yet...")


    def fix_sort(self):
        """
        Reset atom IDs and corresponding bond IDs for plumed.
        """
        # sort the atom in case they arent already
        self.atoms.sort(key=lambda atom: atom.atom_id)
        for index, atom in enumerate(self.atoms):
            if index == 0:
                pass
            else:
                previous_atom = self.atoms[index - 1]
                if atom.atom_id - previous_atom.atom_id != 1:
                    atom.atom_id = previous_atom.atom_id + 1
        
        # check if something went wrong
        for atom in self.atoms:
            if atom.atom_id > len(self.atoms):
                print("Something went wrong during fix_sort!!!!!!!!!")
                print(atom)
                sys.exit(1)


    def remove_bond(self, bond: Bond):
        self.bonds.remove(bond)
        self.header.bonds -= 1
        self.header.bond_types -= 1

        id1, id2 = bond.atom1.atom_id, bond.atom2.atom_id
        
        atoms_map = {atom.atom_id : atom for atom in self.atoms}
        atoms_map[id1].bonded.remove(id2)
        atoms_map[id2].bonded.remove(id1)


    def set_angle_coeff(self, value: float):
        """Sets a single coefficient for all angles in the network

        Parameters
        ----------
        value : float
            _description_
        """
        if self.angles:
            for angle in self.angles:
                angle.energy = value
        else:
            raise Exception("No angle data present")


    def set_source_target(
        self,
        source_beads: tuple[int, int],
        target_beads: tuple[int, int],
        overwrite_existing: bool = True,
        source_beads_mass: float = 100000000.0,
        target_beads_mass: float = 1.0
    ):
        """
        Changes the types of atoms of source and target beads from 1
        to 2 and 3, respectively. Assigns both a default mass.
        If `overwrite_exisiting` is True, changes the previous source and target beads
        to normal.
        """
        atoms_map = {atom.atom_id: atom for atom in self.atoms}
        if overwrite_existing is True:
            for atom in self.atoms:
                atom.atom_type = 1
        # source beads
        atoms_map[source_beads[0]].atom_type = 2
        atoms_map[source_beads[1]].atom_type = 2
        self.masses[2] = source_beads_mass
        # target beads
        atoms_map[target_beads[0]].atom_type = 3
        atoms_map[target_beads[1]].atom_type = 3
        self.masses[3] = target_beads_mass
        # fix header
        n_atom_types: int = len(set([atom.atom_type for atom in self.atoms]))
        self.header.atom_types = n_atom_types
        self.masses = {1: 1.0, 2: source_beads_mass, 3: target_beads_mass}

    @classmethod
    def from_atoms(
        cls,
        input_file: str,
        periodic: bool = True,
        include_default_masses: float = 1.0,
        include_angles: bool = True,
        include_dihedrals: bool = True,
        zero_z: bool = True,
    ) -> Network:
        """
        Reads the lammps data file with only atoms present.
        Returns a `Network` object.

        Arguments:

        - `input_file` : str - path to the lammps data file
        - `include_angles` : bool - whether to include (read or calculate) angles
        - `inclue_dihedrals` : bool - whether to include (read or calculate) dihedrals
        - `zero_z` : bool - whether to set the z coordinates to zero (for 2D networks)
        - `include_default_masses` : int - whether to include masses for atom.
                                           0 to skip.
        """
        with open(input_file, "r", encoding="utf8") as f:
            content = f.readlines()

        box = Box.from_data_file(input_file)
        atoms = get_atoms(content)
        bonds = make_bonds(atoms, box, periodic=periodic)

        # we assume that there's at least one dangling bead
        # if not, nothing bad will happen anyway
        dangling_beads: int = 1
        steps: int = 1
        while dangling_beads > 0:
            atoms, dangling_beads = delete_dangling(atoms)
            bonds = make_bonds(atoms, box, periodic)
            steps += 1

        if zero_z:
            for atom in atoms:
                atom.z = 0.0


        angles = Network._compute_angles(atoms, box) if include_angles else []
        if include_default_masses > 0:
            # print("Assigning default mass of 1.0 to all atom types.")
            masses = {1: include_default_masses}
        else:
            masses = None

        header = Header(atoms, bonds, box, angles=angles)
        return Network(atoms, bonds, box, header, angles=angles, masses=masses)

    @classmethod
    def from_data_file(
        cls,
        input_file: str,
        include_default_masses: float = 1.0,
        include_angles: bool = True,
        include_dihedrals: bool = True,
        zero_z: bool = True,
    ) -> Network:
        """
        Reads the lammps data file and returns a `Network` object.

        Arguments:

        - `input_file` : str - path to the lammps data file
        - `include_angles` : bool - whether to include (read or calculate) angles
        - `inclue_dihedrals` : bool - whether to include (read or calculate) dihedrals
        - `zero_z` : bool - whether to set the z coordinates to zero (for 2D networks)
        - `include_default_masses` : bool - whether to include the default the default
                                            mass of 1 unit for all atom types
        """
        box = Box.from_data_file(input_file)

        with open(input_file, "r", encoding="utf8") as f:
            content = f.readlines()

        header_contents = {
            "atoms": 0,
            "bonds": 0,
            "angles": 0,
            "dihedrals": 0,
            "impropers": 0,
            "atom types": 0,
            "bond types": 0,
            "angle types": 0,
            "dihedral types": 0,
            "improper types": 0,
        }
        # first line is reserved for additional info
        # second line in blank
        for line in content[2:]:
            if line.isspace():
                break
            prop = " ".join(line.strip().split()[1:])
            if prop not in header_contents:
                break
            else:
                value = int(line.strip().split()[0])
                header_contents[prop] = value

        n_atoms = header_contents["atoms"]
        n_atom_types = header_contents["atom types"]
        n_bonds = header_contents["bonds"]
        n_angles = header_contents["angles"]
        n_dihedrals = header_contents["dihedrals"]

        atoms = []
        bonds = []
        angles = []
        dihedrals = []  # noqa: F841

        location: dict[str, tuple[int | None, ...]] = {
            "atoms": tuple(),
            "bonds": tuple(),
            "angles": tuple(),
            "dihedrals": tuple(),
            "masses": tuple(),
            "bond_coeffs": tuple(),
            "angle_coeffs" : tuple(),
        }
        atoms_start      : int | None = None
        atoms_end        : int | None = None
        bonds_start      : int | None = None
        bonds_end        : int | None = None
        angles_start     : int | None = None
        angles_end       : int | None = None
        bond_coeffs_start: int | None = None
        bond_coeffs_end  : int | None = None
        angle_coeffs_start: int | None = None
        angle_coeffs_end: int | None = None
        masses_start     : int | None = None
        masses_end       : int | None = None

        for index, line in enumerate(content):
            if "Atoms" in line.strip():
                atoms_start = index + 2
                atoms_end = atoms_start + n_atoms
                location["atoms"] = (atoms_start, atoms_end)
            if "Bonds" in line.strip():
                bonds_start = index + 2
                bonds_end = bonds_start + n_bonds
                location["bonds"] = (bonds_start, bonds_end)
            if "Bond Coeffs" in line.strip():
                bond_coeffs_start = index + 2
                bond_coeffs_end = bond_coeffs_start + n_bonds
                location["bond_coeffs"] = (bond_coeffs_start, bond_coeffs_end)
            if "Angles" in line.strip():
                angles_start = index + 2
                angles_end = angles_start + n_angles
                location["angles"] = (angles_start, angles_end)
            if "Angle Coeffs" in line.strip():
                angle_coeffs_start = index + 2
                angle_coeffs_end = angle_coeffs_start + n_angles
                location["angle_coeffs"] = (angle_coeffs_start, angle_coeffs_end)
            if "Dihedrals" in line.strip():
                dihedrals_start = index + 2
                dihedrals_end = dihedrals_start + n_dihedrals
                location["dihedrals"] = (dihedrals_start, dihedrals_end)
            if "Masses" in line.strip():
                masses_start = index + 2
                masses_end = masses_start + n_atom_types
                location["masses"] = (masses_start, masses_end)

        if location["atoms"]:
            assert(atoms_start is not None)
            for line in content[atoms_start:atoms_end]:
                data = line.split()
                atom_id = int(data[0])
                atom_type = int(data[2])
                x_coord = float(data[4])
                y_coord = float(data[5])
                z_coord = 0 if zero_z else float(data[6])
                atoms.append(Atom(atom_id, 0.0, x_coord, y_coord, z_coord, atom_type=atom_type))
        else:
            print("[ERROR]: Something went wrong when trying to read atoms from the file.")

        if location["bonds"] and location["bond_coeffs"]:
            assert(bonds_start is not None)
            assert(bond_coeffs_start is not None)

            atoms_map = {atom.atom_id: atom for atom in atoms}
            for bond_line, bond_coeff_line in zip(
                content[bonds_start:bonds_end],
                content[bond_coeffs_start:bond_coeffs_end],
            ):
                data = bond_line.split()
                atom1_id = int(data[2])
                atom2_id = int(data[3])
                atom1 = atoms_map[atom1_id]
                atom2 = atoms_map[atom2_id]
                atom1.bonded.append(atom2_id)
                atom2.bonded.append(atom1_id)

                # reading bond length and coeff from file here to avoid
                # unrealistic bond lengths in case the network is periodic
                bond = Bond(atom1, atom2)
                bond.length = float(bond_coeff_line.split()[2])
                bond.bond_coefficient = float(bond_coeff_line.split()[1])
                bonds.append(bond)
        else:
            print("[ERROR]: Something went wrong when reading bonds from the file.")

        # at this point, the bare minumum for the network sould be present
        header = Header(atoms, bonds, box, atom_types=n_atom_types)
        local_network = Network(atoms, bonds, box, header)

        if location["angles"] and location["angle_coeffs"]:
            assert(angles_start is not None)
            assert(angle_coeffs_start is not None)
            
            if include_angles is True:
                atoms_map = {atom.atom_id: atom for atom in atoms}
                for angle_line, angle_coeff_line in zip(
                    content[angles_start:angles_end],
                    content[angle_coeffs_start:angle_coeffs_end],
                ):
                    data = angle_line.split()
                    angle_id = int(data[0])
                    atom1 = atoms_map[int(data[2])]
                    atom2 = atoms_map[int(data[3])]
                    atom3 = atoms_map[int(data[4])]
                    angle_coeff = float(angle_coeff_line.split()[1])
                    angle_value = float(angle_coeff_line.split()[2])
                    angles.append(Angle(angle_id, atom1, atom2, atom3, box, angle_coeff, angle_value))
                    
                local_network.angles = angles
                header.angles = len(angles)
                header.angle_types = len(angles)
            else:
                pass
        else:
            #print("No angle data have been found")
            if include_angles is True:
                # print("Calculating angles..")
                angles = Network._compute_angles(atoms, box)
                local_network.angles = angles
                header.angles = len(angles)
                header.angle_types = len(angles)
                # print(f"Angles calculated: {len(angles)}")
            else:
                #print("Angles are not included")
                pass
        if location["dihedrals"]:
            # TODO Implement reading and writing dihedrals
            #print(f"Dihedrals expected: {n_dihedrals}")
            if include_dihedrals is True:
                #print("Dihedrals are not yet emplemented.")
                pass
            else:
               # print("Dihedrals are not included")
               pass
        else:
            #print("No dihedrals data have been found")
            if include_dihedrals is True:
                # print("Dihedrals are not yet emplemented.")
                pass
            else:
                #print("Dihedrals are not included")
                pass

        if location["masses"]:
            masses = {}
            for line in content[masses_start:masses_end]:
                data = line.split()
                masses[int(data[0])] = float(data[1])

            #print("Found mass info: ")
            #for key, value in masses.items():
                #print(f"    Atom type: {key}, mass: {value} units")
            local_network.masses = masses
        else:
           # print("No masses data have been found")
            if include_default_masses != 0:
                # print(f"Assigning default mass of {include_default_masses} to all atom types.")
                masses = {
                    atom_type: include_default_masses
                    for atom_type in range(1, n_atom_types + 1)
                }
                local_network.masses = masses

        return local_network

    @classmethod
    def from_pickle(cls, filepath: str) -> Network:
        with open(filepath, 'rb') as network_file:
            return pickle.load(network_file)


    def write_to_file(self, target_file: str, add_box_line: bool = True) -> str:
        """
        Writes network to a file a returns the path
        """
        path = os.path.abspath(os.path.join(os.getcwd(), target_file))
        with open(path, "w", encoding="utf8") as file:
            self.header.write_header(file, add_box_line=add_box_line)

            if self.masses:
                file.write("\nMasses # ['atom_id', 'mass']\n\n")
                for key, value in self.masses.items():
                    file.write(" ".join([str(key), str(float(value))]) + "\n")
            else:
                print("Atomic masses are not specified.")

            if self.atoms:
                # write `Atoms` section
                # 7-7-7-11-11-11-11
                legend = ["atomID", "moleculeID", "atomType", "charge", "x", "y", "z"]
                file.write(f"\nAtoms # {legend}\n\n")
                for atom in self.atoms:
                    properties = [
                        atom.atom_id,
                        1,  # molecule ID. always 1 for now
                        atom.atom_type,  # defaults to 1 when construsted
                        format(0.0, ".6f"),  # charge. always neutral for now
                        format(round(atom.x, 9), ".9f"),
                        format(round(atom.y, 9), ".9f"),
                        format(round(atom.z, 9), ".9f"),
                    ]
                    widths = [7, 7, 7, 11, 11, 11, 11]
                    line = table_row(properties, widths)
                    file.write(line)

            if self.bonds:
                # write `Bond Coeffs` section
                # 7-11-11
                legend = ["bondID", "bondCoeff", "d"]
                file.write(f"\nBond Coeffs # {legend}\n\n")
                for n, bond in enumerate(self.bonds):
                    properties = [
                        n + 1,
                        format(round(bond.bond_coefficient, 9), ".9f"),
                        format(round(bond.length, 9), ".9f"),
                    ]
                    widths = [7, 25, 25]
                    line = table_row(properties, widths)
                    file.write(line)

                # write `Bonds` section
                # 10-10-10-10
                legend = ["ID", "type", "atom1", "atom2"]
                file.write(f"\nBonds # {legend}\n\n")
                for _id, bond in enumerate(self.bonds):
                    properties = [
                        _id + 1,
                        _id + 1,
                        bond.atom1.atom_id,
                        bond.atom2.atom_id,
                    ]
                    widths = [10, 10, 10, 10]
                    line = table_row(properties, widths)
                    file.write(line)

            if self.angles:
                # write `Angle Coeffs` section
                # 7-11-11
                legend = ["angleID", "energy", "value (deg)"]
                file.write(f"\nAngle Coeffs # {legend}\n\n")
                for angle in self.angles:
                    properties = [
                        angle.angle_id,
                        format(angle.energy, ".6f"),
                        format(angle.value, ".6f"),
                    ]
                    widths = [7, 11, 11]
                    line = table_row(properties, widths)
                    file.write(line)

                # write `Angles` section
                # 10-10-10-10-10
                legend = ["angleID", "angleType", "atom1", "atom2", "atom3"]
                file.write(f"\nAngles # {legend}\n\n")
                for angle in self.angles:
                    properties = [
                        angle.angle_id,
                        angle.angle_id,  # type the same as id
                        angle.atom1.atom_id,
                        angle.atom2.atom_id,
                        angle.atom3.atom_id,
                    ]
                    widths = [10, 10, 10, 10, 10]
                    line = table_row(properties, widths)
                    file.write(line)

            if self.dihedrals:
                # print("Writing dihedrals to a file is not implemented.")
                pass
                # TODO Implement reading and writing dihedrals

        if os.path.exists(path) and os.path.getsize(path) > 0:
            #print(f"Output was written in: {os.path.abspath(path)}")
            return path
        print(f"Problem saving network to {path}")
        sys.exit(1)


    def to_pickle(self, filepath: str):
        with open(filepath, 'wb') as network_file:
            pickle.dump(self, network_file)


def get_atoms(file_contents: list[str]) -> list[Atom]:
    n_atoms = 0
    atoms_start_line = 0
    atoms_end_line = 0
    atoms = []

    for i, line in enumerate(file_contents[:20]):
        # skip the comment lines
        if line[0] == "#":
            continue
        # get rid the inline comments
        if "#" in line:
            line = " ".join(line.split()[: line.find("#")])
        # read number of atoms at the top of data file
        if "atoms" in line.split():
            n_atoms = int(line.split()[0])
        # find the Atoms part
        if "Atoms" in line.split():
            atoms_start_line = i + 2
            atoms_end_line = atoms_start_line + n_atoms
            break
    # Go line-by-line extracting useful info
    for atom_line in file_contents[atoms_start_line:atoms_end_line]:
        atom_id: int = int(atom_line.strip().split()[0])
        disk_type: int = int(atom_line.strip().split()[1])
        atom_diamater = float(atom_line.strip().split()[2]) 
        x: float = float(atom_line.split()[4])
        y: float = float(atom_line.split()[5])
        z: float = float(atom_line.split()[6])
        atoms.append(Atom(atom_id, atom_diamater, x, y, z, disk_type=disk_type))
    return atoms


def delete_dangling(atoms: list[Atom]) -> tuple[list, int]:
    new_atoms = [atom for atom in atoms if atom.n_bonds > 2]
    difference = len(atoms) - len(new_atoms)
    for atom in new_atoms:
        # erase information about the number of bonds and bonded neighbour ids
        atom.n_bonds = 0
        atom.bonded = []
    return (new_atoms, difference)


def make_surrounding(atoms: list[Atom], box: Box, dimensions: int = 2) -> list[Atom]:
    surrounding_atoms = set()
    # spawn neighboring atoms along the x and y axis of the bounding box, 8 in total
    if dimensions == 2:
        for atom in atoms:
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    if not x * box.x == y * box.y == 0:
                        surrounding_atoms.add(deepcopy(atom).translate(box, (x, y, 0)))
    elif dimensions == 3:
        for atom in atoms:
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    for z in (-1, 0, 1):
                        if not x * box.x == y * box.y == z * box.z == 0:
                            surrounding_atoms.add(
                                deepcopy(atom).translate(box, (x, y, z))
                            )
    else:
        raise NotImplementedError(
            "Allowed values for `dimensions` argument are either 2 or 3."
        )
    return list(surrounding_atoms)


def make_bonds(atoms: list[Atom], box: Box, periodic: bool) -> list:
    # first make all the bond inside the simulation box
    bonds = set()
    for atom_k in atoms:
        for atom_j in atoms:
            if atom_k != atom_j:
                if atom_k.dist(atom_j) <= (
                    (atom_k.diameter / 2) + (atom_j.diameter / 2)
                ):
                    bonds.add(Bond(atom_k, atom_j))
                    atom_k.bonded.append(atom_j.atom_id)
                    atom_k.n_bonds += 1

    extra_bonds = set()
    if periodic:
        atoms_map = {atom.atom_id : atom for atom in atoms}
        edge_atom = [atom for atom in atoms if atom.on_edge(box, 2.0)]
        neighbours = make_surrounding(atoms, box)
        edge_neighbors = [atom for atom in neighbours if atom.on_edge(box, 2.0)]

        for main_atom in edge_atom:
            for outside_atom in edge_neighbors:
                if main_atom.dist(outside_atom) <= (
                    (main_atom.diameter / 2) + outside_atom.diameter / 2
                ):
                    bond = Bond(main_atom, outside_atom)
                    bond.atom2 = atoms_map[outside_atom.atom_id]
                    extra_bonds.add(bond)
                    main_atom.bonded.append(outside_atom.atom_id)
                    main_atom.n_bonds += 1

    return list(bonds.union(extra_bonds))


if __name__ == "__main__":
    usage_info = "\n[USAGE]:\n\n    ./network.py target_file [OPTIONAL] out_file.\n"
    if len(sys.argv) < 2:
        print("\n[ERROR]: target file was not provided.")
        print(usage_info)
        sys.exit(0)
    elif sys.argv[1] == "help":
        print(usage_info)
        sys.exit(0)

    input_file_path = sys.argv[1]
    print(f"Input file: {os.path.abspath(input_file_path)}")

    if len(sys.argv) > 2:
        out_file_name = sys.argv[2]
        out_file_path = os.path.join(os.path.dirname(os.path.abspath(input_file_path)), out_file_name)
    else:
        input_file_path = os.path.abspath(input_file_path)
        input_dir = os.path.dirname(input_file_path)
        input_file_name = os.path.basename(input_file_path).split(".")[0]
        out_file_name = "".join((input_file_name, "_out.lmp"))
        out_file_path = os.path.join(input_dir, out_file_name)

    # constructing the bare minimum network from atomic coordinates
    network = Network.from_atoms(input_file_path)

    network.write_to_file(out_file_path)
