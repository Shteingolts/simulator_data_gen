import os
import shutil
import subprocess
from copy import deepcopy
from torch_geometric.data import Data
from perturber_model import PerturbSimulate, ModelInputs, actual_dists
import torch 
import numpy as np
import random 
from vel_recalc_model import Model
from utils import load_data, adjust_box_for_graph
from convert import network_from_data

# run lammps and save network
def run(data, diff_optimizer, random_simulations: list[int], path: str, dir: str, rand=True, org=False, fixed=False, device='cuda'):
    # set working directory to path 
    os.chdir(f'{path}/{dir}') 

    for sim in random_simulations:
        # define directory name 
        if rand: 
            name = f'network_rand_{sim}'
        elif org:
            name = f'network_org_{sim}'
        elif fixed:
            name = f'network_fixed_{sim}'
        else: 
            name = f'network_test_{sim}'
    
        # make directory with the name 
        os.makedirs(f'{os.getcwd()}/{name}', exist_ok=True)
        os.chdir(f'{os.getcwd()}/{name}')
        # import the LAMMPS files 
        shutil.copyfile(f'{path}/test/init.mod', f'{os.getcwd()}/init.mod')
        shutil.copyfile(f'{path}/test/compress.deformation', f'{os.getcwd()}/compress.deformation')
        shutil.copyfile(f'{path}/test/potential.mod', f'{os.getcwd()}/potential.mod')
        # get a sample 
        to_test= deepcopy(data[sim][1].to(device))
        if rand: 
            new_nodes = to_test.x + (torch.rand(to_test.x.shape)/2).to(device) * torch.randint(0, 2,size=to_test.x.shape).to(device)
            tmp = Data(x= new_nodes, edge_index=to_test.edge_index, edge_attr=to_test.edge_attr, box=to_test.box, atom_ids=to_test.atom_ids) 
            new_edges = actual_dists(tmp)
            new = Data(x=new_nodes,edge_index=to_test.edge_index,edge_attr=new_edges,box = to_test.box, atom_ids=to_test.atom_ids)
        elif org: 
            new = Data(x=to_test.x,edge_index=to_test.edge_index,edge_attr=to_test.edge_attr,box = to_test.box, atom_ids=to_test.atom_ids)
        elif fixed:
            # copy network
            new_inps = ModelInputs(data[sim][0].to(device), data[sim][1].to(device), data[sim][2].to(device))
            new = diff_optimizer(new_inps)
            new = adjust_box_for_graph(data[sim][1].cpu().detach(), new)
        else:
            new_inps = ModelInputs(data[sim][0].to(device), data[sim][1].to(device), data[sim][2].to(device))
            new = diff_optimizer(new_inps)
    
        to_save = network_from_data(new)
        print(to_save)
        to_save.write_to_file("network.lmp")
        # run lammps 
        subprocess.run('lmp -in compress.deformation', shell=True) 
        # get the Y values using the logfile and save it in an array 
        os.chdir(f'{path}/{dir}')

def clean(path):
    # set path
    os.chdir(f'{path}')
    # remove directories 
    subprocess.run('rm -rf test_rand', shell=True) 
    subprocess.run('rm -rf test_networks', shell=True)
    subprocess.run('rm -rf org_networks', shell=True) 
    subprocess.run('rm -rf test_fixed_networks', shell=True) 

    # create new, empty ones
    subprocess.run('mkdir test_rand', shell=True) 
    subprocess.run('mkdir test_networks', shell=True)
    subprocess.run('mkdir org_networks', shell=True) 
    subprocess.run('mkdir test_fixed_networks', shell=True) 
    print('done')




if __name__ == "__main__":
    # load the data
    data = load_data("data_pt/data_2.5_coord.pt", skip=10)
    # define inputs 
    device = 'cuda' if torch.cuda.is_available() else 'mps' 
    # define simulator
    simulator = Model(data[0][0], hidden_size=128,n_layers=1).to(device)
    simulator.load_state_dict(torch.load('simulator.pt', map_location=device))
    model_inps = ModelInputs(data[0][0], data[0][1], data[0][2])
    # define the optimizer
    diff_optimizer = PerturbSimulate(model_inps, simulator, 128).to(device)
    diff_optimizer.load_state_dict(torch.load("model_opt.pt", map_location=device))
    # pick 10 random simulations from the data 
    random_simulations = list(np.random.choice(range(0, len(data) - 1), 40, replace=False)) #[random.randint(0,len(data) - 1) for i in range(10)]
    # Write simulation metadata into a file
    file = open("rand_sims.txt", "w+")
    content = str(random_simulations)
    file.write(content)
    file.close()
    path = "/home/sergey/python/simulator_data_gen"

    

    # clean 
    clean(path)
    # generate data
    run(data, diff_optimizer, random_simulations, path, dir = 'test_networks', rand=False, device=device) # optimized
    run(data, diff_optimizer, random_simulations, path, dir = 'test_fixed_networks', rand=False, fixed=True, device=device) # optimized + fixed box
    run(data, diff_optimizer, random_simulations, path, dir = 'test_rand', device=device) # random 
    run(data, diff_optimizer, random_simulations, path, dir = 'org_networks', rand=False, org=True, device=device) # original  

