import os
import sys
import shutil

path = "/home/sergey/work/simulator_data_gen/data/raw/200_networks"

for subdir in os.listdir(path):
    subpath = os.path.join(path, subdir)
    for file in os.listdir(subpath):
        if file.startswith("proc_") or file.startswith("comp_"):
            print(f"Removing {os.path.join(subpath, file)}")
            shutil.rmtree(os.path.join(subpath, file))
