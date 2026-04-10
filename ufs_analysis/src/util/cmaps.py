# ---------------------------------------------------------------------------------------------------------------------
#  Filename: cmaps.py
#  Created by: Tariq Hamzey, Cristiana Stan
#  Created on: 26 March, 2026
#  Purpose: Read in custom color maps yaml file, convert it to matplotlib-ready Python dictionary.
# ---------------------------------------------------------------------------------------------------------------------

import os
import yaml


def process_cmaps_yaml():

    # Current working directory
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Get source notebook directory
    cmaps_file_path = os.path.join(cwd, 'cmaps.yml')

    # Load cmaps file
    with open(cmaps_file_path, 'r') as file:
        cmap_dict = yaml.safe_load(file)

    # Instantiate the processed cmap which will be a Python dict of lists of rgb tuples
    processed_cmap = dict.fromkeys(cmap_dict.keys())

    # Iterate over cmaps and build the dictionary
    for cmap in cmap_dict.keys():
        this_cmap = cmap_dict[cmap]

        # Instantiate rgb list
        processed_cmap[cmap] = []

        for i in range(len(this_cmap)):
            this_r = this_cmap[i][0] / 255
            this_g = this_cmap[i][1] / 255
            this_b = this_cmap[i][2] / 255

            # Add this rgb to list
            processed_cmap[cmap].append((this_r, this_g, this_b))

    return processed_cmap
