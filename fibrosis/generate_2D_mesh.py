"""

Åshild Telle / University of Washington / 2023

Script for removing cells from the geometry, replicating the progression of cardiac fibrois.

"""

import numpy as np
import dolfin as df
from mpi4py import MPI
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from random import choices

from emimechanicalmodel import (
    load_mesh,
    compute_active_component,
    EMIModel,
)

from collagen_alignment import assign_collagen_distribution
from plot_collagen import plot_collagen_distribution

from setup_replacement_fibrosis import (
    find_middle_cells_transversely,
    find_middle_cells_longitudinally,
    find_random_cells,
    replace_cells_with_matrix,
)

parser = ArgumentParser()

parser.add_argument(
    "-r",
    "--remove_type",
    default="longitudinally",
    type=str,
    choices=["longitudinally", "transversely", "randomly"],
    help="Remove cells along ... direction",
)

parser.add_argument(
    "-n",
    "--remove_num_cells",
    default=0,
    type=int,
    help="How many cells to remove",
)


parser.add_argument(
    "-sd",
    "--seed",
    default=0,
    type=int,
    help="Random seed (for 'random' distribution)",
)


pp = parser.parse_args()

remove_type, remove_num_cells, seed = (
    pp.remove_type,
    pp.remove_num_cells,
    pp.seed
)


# load mesh, subdomains

num_cells_xdir = 2 #8
num_cells_ydir = 3 #16
#mesh_file = f"meshes/mesh_6x12_cells.h5"
mesh_file = f"meshes/2d_mesh_{num_cells_xdir}_{num_cells_ydir}_5.h5"
mesh, volumes = load_mesh(mesh_file)

if remove_type == "longitudinally":
    print(f"Removing {remove_num_cells} cells longitudinally")
    remove_cells = find_middle_cells_longitudinally(mesh, volumes, num_cells_xdir, remove_num_cells, 1.6, seed)
elif remove_type == "transversely":
    print(f"Removing {remove_num_cells} cells transversely")
    remove_cells = find_middle_cells_transversely(mesh, volumes, num_cells_ydir, remove_num_cells, 2.8, seed)
elif remove_type == "randomly":
    print(f"Removing {remove_num_cells} cells randomly")
    remove_cells = find_random_cells(mesh, volumes, remove_num_cells, seed)
else:
    raise NotImplementedError

if len(remove_cells) > 0:
    replace_cells_with_matrix(mesh, volumes, remove_cells)

colors = ["tab:blue", "tab:purple", "tab:red"]

for N, color in zip([10], colors):
    for kappa in [0.0]:
        fout = f"meshes/mesh_fibrosis_{remove_type}_{remove_num_cells}_{seed}.h5"
        #fout = f"meshes/mesh_baseline_with_collagen.h5"
        print(fout, kappa, N)
        #plot_collagen_distribution(mesh, color=color, mu=0, kappa=kappa, N=N)

        collagen = assign_collagen_distribution(mesh, mu=0, kappa=kappa, N=N)
        
        with df.HDF5File(mesh.mpi_comm(), fout, 'w') as out:
            out.write(mesh, 'mesh/')
            out.write(volumes, 'subdomains/')
            out.write(collagen, 'collagen_dist/')

