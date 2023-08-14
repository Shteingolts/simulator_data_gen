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
thermo_modify norm no
