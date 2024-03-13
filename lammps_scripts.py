import os
import random


class TemperatureRange:
    """Simple data class to store temperature range information."""

    T_start: float
    T_end: float
    bias: float

    def __init__(
        self, T_start: float = 0.0001, T_end: float = 0.0001, bias: float = 6.0
    ) -> None:
        self.T_start = T_start
        self.T_end = T_end
        self.bias = bias

    def __repr__(self) -> str:
        return f"TempRange: start={self.T_start}, stop={self.T_end}, bias={self.bias}"


class LJSimulation:
    """Defines the initial lammps calculation which is used to generate a disordered network.
    Default values work 100%, anything you change may not. The length of `atom_sizes` array
    needs to be equal to `n_atom_types` (yes, the latter attribute is useless, i know).

    Raises
    ------
    IOError
        Raises an error if `len(atom_sizes) != n_atom_sizes`
    """

    n_atoms: int
    n_atom_types: int
    atom_sizes: list[float]
    box_dim: list[float]
    temperature_range: TemperatureRange = TemperatureRange()
    n_steps: int

    def __init__(
        self,
        n_atoms: int = 200,
        n_atom_types: int = 4,
        atom_sizes: list[float] = [1.6, 1.4, 1.2, 1.0],
        box_dim: list[float] = [-7.0, 7.0, -7.0, 7.0, -0.1, 0.1],
        temperature_range: TemperatureRange = TemperatureRange(),
        n_steps: int = 30000,
    ):
        if len(atom_sizes) != n_atom_types:
            raise IOError(
                "[ERROR] Length of `atom_sizes` should be equal to `n_atom_types`"
            )

        self.n_atoms = n_atoms
        self.n_atom_types = n_atom_types
        self.atom_sizes = atom_sizes
        self.box_dim = box_dim
        self.temperature_range = temperature_range
        self.n_steps = n_steps

    def write_to_file(self, directory: str | None = None):
        create_atoms_section: str = ""
        group_section: str = ""
        group_size_section: str = ""
        for i in range(self.n_atom_types):
            create_atoms_section += f"create_atoms {i+1} random ${{npart}} {random.randint(1, 999999)} NULL overlap 0.5 maxtry 100\n"
            group_section += f"group size{i+1} type {i+1}\n"
            group_size_section += f"set group size{i+1} diameter {self.atom_sizes[i]}\n"

        script_template: str = f"""
#area fraction
# 0.85 is in the liquid state at T=1
# 0.90 is a crystalline solid)
variable afrac equal 1.00

#number of particles in 20x20 area
variable npart  equal ${{afrac}}*{self.n_atoms // self.n_atom_types}


#we use the LJ potential epsilon as energy scale,
#and sigma as length scale.
units		 lj
dimension    2
atom_style	 sphere 
boundary     p p p
neighbor     0.5   bin
neigh_modify every 1 delay 0 check yes


# create 2D box
region box block {self.box_dim[0]} {self.box_dim[1]} {self.box_dim[2]} {self.box_dim[3]} {self.box_dim[4]} {self.box_dim[5]}
create_box {self.n_atom_types} box


# put z=0 all the time
fix 2d  all enforce2d
comm_modify vel yes


# put the particles randomly into the box
{create_atoms_section}


# create groups named size1 to size4 of 4 types of atoms defined above
{group_section}


# sets the sizes of atoms within the defined groups
{group_size_section}


# define soft granualr potential forces (as done in Reid paper).
pair_style gran/hooke 1.0 0.0 0.0 0.0 0.0 0
pair_coeff * *

# minimize energy first to avoid overlapping particles
minimize 1e-10 1e-10 1000 1000
#fix 10 all box/relax aniso ${{Pres}} vmax 0.001


reset_timestep 0
fix 10 all npt/sphere temp {self.temperature_range.T_start} {self.temperature_range.T_end} {self.temperature_range.bias} iso 0.1 0.1 0.6 disc


# output time step, temperature, average kinetic and potential energy
# thermo_style custom step temp ke pe
# thermo		100
dump            1 all atom 10 dump.lammpstrj

#time step of integrator
timestep	0.006

#number of integration steps to run
run		{self.n_steps}

write_data coord.dat
"""

        if directory is None:
            with open("lammps.in", "w", encoding="utf8") as lammps_file:
                lammps_file.write(script_template)
        else:
            path = os.path.join(directory, "lammps.in")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf8") as lammps_file:
                lammps_file.write(script_template)


class CompressionSimulation:
    """Defines the compression simulation which is used to generate data for GNN simulator."""

    network_filename: str
    strain: float
    strain_rate: float
    temperature_range: TemperatureRange
    dump_frequency: int

    def __init__(
        self,
        network_filename: str = "network.lmp",
        strain: float = 0.05,
        strain_rate: float = 1e-5,
        temperature_range: TemperatureRange = TemperatureRange(),
        dump_frequency: int | None = None,
    ) -> None:
        self.network_filename = network_filename
        self.strain = strain
        self.strain_rate = strain_rate
        self.temperature_range = temperature_range
        self.dump_frequency = (
            int(strain / strain_rate * 100 / 2000) if dump_frequency is None else 1000
        )

    def write_to_file(self, directory: str | None = None):
        deformation_script_template: str = f"""
variable dt equal 0.01
variable strain equal {self.strain}
variable srate equal {self.strain_rate}
variable M equal 1000 # averaging time for properties
variable strainsteps equal ${{strain}}/${{dt}}/${{srate}}


dimension 2
include init.mod
include potential.mod


#change_box all triclinic
comm_modify mode single cutoff 10.0 vel yes
atom_modify sort 0 0


# Compute initial state
fix 3 all box/relax x 0 y 0 # xy 0  #MS comment: relax if the network is periodic
minimize ${{etol}} ${{ftol}} ${{maxiter}} ${{maxeval}}
thermo 100
# thermo_style custom step atoms temp press c_vx c_vy c_vz
unfix 3
fix 2d all enforce2d


# These formulas define the derivatives w.r.t. strain components
# Constants uses $, variables use v_ 
timestep        ${{dt}}
variable nsteps equal round(${{strainsteps}})

#fix             4 all nve
#fix		     6 all press/berendsen y  0.0 0.0 1 x 0.0 0.0 10
#fix             6 all npt temp 0.0001 0.0001 1000.0 y 0.0 0.0 1000.0
fix             5 all langevin {self.temperature_range.T_start} {self.temperature_range.T_end} {self.temperature_range.bias} 904297
fix             6 all nph y 0.0 0.0 1000.0 ptemp 0.0001
fix     	     9 all deform 1 x erate -${{srate}} units box remap x

# For PLUMED2
# fix 22 all plumed plumedfile plumed.dat outfile plumed.out

#To avoid flying ice cube phenomenon:
fix 21 all momentum 100 linear 1 1 1 angular

dump 1 all custom {self.dump_frequency} dump.lammpstrj id x y z vx vy vz
run  ${{nsteps}}
"""
        init_script_template: str = f"""
# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
# variable up equal 1.0e-1
 
# Define the amount of random jiggle for atoms
# This prevents atoms from staying on saddle points
# variable atomjiggle equal 1.0e-6

# Uncomment one of these blocks, depending on what units
# you are using in LAMMPS and for output

# metal units, elastic constants in eV/A^3
#units		metal
#variable cfac equal 6.2414e-7
#variable cunits string eV/A^3

# Choose potential
# metal units, elastic constants in GPa
units		metal
atom_style	full
#pair_style      lj/cut 1.0 
#pair_style      yukawa 1.0 2.0  
#pair_modify     mix arithmetic
bond_style      harmonic
angle_style     harmonic
dihedral_style  zero
special_bonds   amber
variable cfac equal 1.0e-4
variable cunits string GPa
boundary	p p p

# Need to set mass to something, just to satisfy LAMMPS
#mass 1 1.0e-20
read_data {self.network_filename}

# real units, elastic constants in GPa
#units		real
#variable cfac equal 1.01325e-4
#variable cunits string GPa

# Define minimization parameters
variable etol equal 0.0 
variable ftol equal 5.0e-10
variable maxiter equal 100000
variable maxeval equal 20000
variable dmax equal 1.0e-2
"""
        potential_script_template: str = """
# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Setup neighbor style
neighbor 3.0 nsq
neigh_modify once no every 1 delay 0 check yes
comm_modify cutoff 2.0

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

timestep	0.001


# Setup output
thermo		500
thermo_style custom step temp pe ebond eangle edihed press pxx pyy pzz pxy pxz pyz lx ly lz vol
# thermo_style custom step temp atoms vx vy vz
thermo_modify norm no
"""

        files = {
            "in.deformation": deformation_script_template,
            "init.mod": init_script_template,
            "potential.mod": potential_script_template,
        }

        for filename, content in files.items():
            if directory is None:
                with open(filename, "w", encoding="utf8") as f:
                    f.write(content)
            else:
                path = os.path.join(directory, filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf8") as f:
                    f.write(content)
