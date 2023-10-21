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

from parameter_setup import setup_monitor

from emimechanicalmodel import (
    load_mesh,
    compute_active_component,
    EMIModel,
)

from setup_replacement_fibrosis import (
    find_middle_cells_transversely,
    find_middle_cells_longitudinally,
    find_random_cells,
    replace_cells_with_matrix,
)

parser = ArgumentParser()

parser.add_argument(
    "-s",
    "--scale",
    default=1,
    type=float,
    help="Scaling parameter for ECM stiffness",
)

parser.add_argument(
    "-r",
    "--remove_type",
    default="longitudinally",
    type=str,
    options=["longitudinally", "transversely", "randomly"]
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

scale, remove_type, remove_num_cells, seed = (
    pp.scale,
    pp.remove_type,
    pp.remove_num_cells,
    pp.seed
)


# load mesh, subdomains

num_cells_xdir = 6
num_cells_ydir = 12
mesh_file = f"meshes/mesh_6x12_cells.h5"
mesh, volumes = load_mesh(mesh_file)

if remove_type == "longitudinally":
    print(f"Removing {remove_num_cells} cells longitudinally")
    remove_cells = find_middle_cells_longitudinally(mesh, volumes, num_cells_xdir, remove_num_cells, 1, seed)
elif remove_type == "transversely":
    print(f"Removing {remove_num_cells} cells transversely")
    remove_cells = find_middle_cells_transversely(mesh, volumes, num_cells_ydir, remove_num_cells, 1, seed)
elif remove_type == "randomly":
    print(f"Removing {remove_num_cells} cells randomly")
    remove_cells = find_random_cells(mesh, volumes, remove_num_cells, seed)
else:
    raise NotImplementedError

if len(remove_cells) > 0:
    replace_cells_with_matrix(mesh, volumes, remove_cells)

fout = f"meshes/mesh_{remove_type}_{remove_num_cells}_{seed}.h5"

with df.HDF5File(mesh.mpi_comm(), fout, 'w') as out:
    out.write(mesh, 'mesh/')
    out.write(volumes, 'subdomains/')
