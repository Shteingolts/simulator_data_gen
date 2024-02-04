class TempRange:
    T_start: float
    T_end: float
    bias: float

    def __init__(
        self, T_start: float = 0.0001, T_end: float = 0.0001, bias: float = 6.0
    ) -> None:
        self.T_start = T_start
        self.T_end = T_end
        self.bias = bias


class LJ_sim:
    n_atoms: int
    n_atom_types: int
    atom_sizes: list[float]
    box_dim: list[int]
    temperature_range: TempRange = TempRange()
    n_steps: int

    def __init__(
        self,
        n_atoms: int = 200,
        n_atom_types: int = 4,
        atom_sizes: list[float] = [1.6, 1.4, 1.2, 1.0],
        box_dim: list[int] = [-7.0, 7.0, -7.0, 7.0, -0.1, 0.1],
        temperature_range: TempRange = TempRange(),
        n_steps: int = 30000,
    ):
        if len(atom_sizes) != n_atom_types:
            raise IOError("[ERROR] Length of `atom_sizes` should be equal to `n_atom_types`")

        self.n_atoms = n_atoms
        self.n_atom_types = n_atom_types
        self.atom_sizes = atom_sizes
        self.box_dim = box_dim
        self.temperature_range = temperature_range
        self.n_steps = n_steps

    def write_to_file(self, path: str = "lammps.in"):
        create_atoms_section: str = ""
        group_section: str = ""
        group_size_section: str = ""
        for i in range(self.n_atom_types):
            create_atoms_section += f"create_atoms {i+1} random ${{npart}} {i+1} NULL overlap 0.5 maxtry 100\n"
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
create_box 4 box


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

        with open(path, "w", encoding="utf8") as lammps_file:
            lammps_file.write(script_template)
