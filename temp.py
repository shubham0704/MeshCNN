import os

dir_path = "./datasets/synthetic_tower_data/sseg"

new_ext = '.seseg'
for file in os.listdir(dir_path):
    file = os.path.join(dir_path,file)
    name, ext = os.path.splitext(file)
    new_name = name + new_ext
    os.rename(file, new_name)
