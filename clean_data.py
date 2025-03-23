import os
import shutil

path = "/home/sergey/work/simulator_data_gen/one_over_l"

for subdir in os.listdir(path):
    if subdir.startswith("data_generation"):
        print(subdir)
    else:
        subpath = os.path.join(path, subdir)
        for file in os.listdir(subpath):
            local_path = os.path.join(subpath, file)
            for random_stuff in os.listdir(local_path):
                if random_stuff.startswith("proc_") or "Ttimes10000" in random_stuff:
                    print(f"Removing {os.path.join(local_path, random_stuff)}")
                    shutil.rmtree(os.path.join(local_path, random_stuff))



# path = "/home/sergey/work/simulator_data_gen/data/raw/200_networks"

# for subdir in os.listdir(path):
#     if subdir.startswith("data_generation"):
#         print(subdir)
#     else:
#         subpath = os.path.join(path, subdir)
#         for file in os.listdir(subpath):
#             # print(file)
#             if file.startswith("comp_1"):
#                 print(f"Removing {os.path.join(subpath, file)}")
#                 shutil.rmtree(os.path.join(subpath, file))
