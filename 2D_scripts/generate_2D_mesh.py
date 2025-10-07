"""

Åshild Telle / University of Washington / 2023

Script for removing cells from the geometry, replicating the progression of cardiac fibrosis.

"""

import numpy as np
import dolfin as df
from mpi4py import MPI
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from random import choices
import matplotlib.pyplot as plt

from emimechanicalmodel import (
    load_mesh,
    compute_active_component,
    EMIModel,
)

def write_collagen_to_file(mesh_file):
    comm = MPI.COMM_WORLD
    h5_file = df.HDF5File(comm, mesh_file, "r")
    mesh = df.Mesh()
    h5_file.read(mesh, "mesh", False)

    collagen_dist = df.MeshFunction("double", mesh, 0)
    
    # this needs to match whatever the subdomain is called in the mesh file
    h5_file.read(collagen_dist, "collagen_dist")

    V = df.FunctionSpace(mesh, "CG", 1)
    theta = df.Function(V)
    theta.vector()[:] = collagen_dist.array()[:]

    name = mesh_file.split(".")[0]
    fid = df.HDF5File(comm, f"{name}_collagen.h5", "w")
    fid.write(theta, "collagen_dist")
    fid.close()
    print("Done!")


from collagen_alignment import assign_collagen_distribution
#from plot_collagen import plot_collagen_distribution

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
    default="replacement",
    type=str,
    choices=["replacement", "interstitial"],
    help="Replacement vs interstitial fibrosis",
)

parser.add_argument(
    "-n",
    "--remove_num_cells",
    default=0,
    type=int,
    help="How many cells to remove",
)

parser.add_argument(
    "-s",
    "--seed",
    default=0,
    type=int,
    help="seed value (for fixed random distribution)",
)


pp = parser.parse_args()

remove_type, remove_num_cells, seed = (
    pp.remove_type,
    pp.remove_num_cells,
    pp.seed
)


# load mesh, subdomains

num_cells_xdir = 8
num_cells_ydir = 15

if remove_type=="replacement":
    mesh_file = "meshes_revisions/baseline.h5"
elif remove_type=="interstitial":
    mesh_file = "meshes_revisions/interstitial.h5"

mesh, volumes = load_mesh(mesh_file)
endomysium_id = 1
perimysium_id = 2

if remove_type=="replacement" and remove_num_cells > 0:
    print(f"Removing {remove_num_cells} cells randomly")
    cell_id_min, cell_id_max = 10, 129

    remove_cells = find_random_cells(mesh, volumes, remove_num_cells, seed, cell_id_min, cell_id_max)
    
    replace_cells_with_matrix(mesh, volumes, remove_cells, perimysium_id)
    
    remove_connections = []
    connection_id = 200
    for i in range(15):
        for j in range(7):
            cell = 10 + i*8 + j
            if (cell in remove_cells or cell+1 in remove_cells) and connection_id not in remove_connections:
                remove_connections.append(connection_id)
            connection_id += 1
        cell = i*8 + 7

    print(remove_connections)    
    replace_cells_with_matrix(mesh, volumes, remove_connections, perimysium_id)


if remove_type=="interstitial" and remove_num_cells > 0:
    print(f"Removing {remove_num_cells} connections randomly")
    cell_id_min, cell_id_max = 200, 304

    remove_cells = find_random_cells(mesh, volumes, remove_num_cells, seed, cell_id_min, cell_id_max)
    replace_cells_with_matrix(mesh, volumes, remove_cells, endomysium_id)

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(4.5, 3))

colors = {0 : "darkgray", 10: "tab:red"}

for N in [10]:
    for kappa in [0, 10]:
        fout = f"meshes_revisions/2d_mesh_{remove_type}_N_{N}_k_{kappa}_seed_{seed}.h5"
        print(fout, kappa, N)

        collagen = assign_collagen_distribution(mesh, volumes, mu=0, kappa_endomysium=kappa, kappa_perimysium=20+kappa, N=N, plot=True, axis_endo=axes[0], axis_peri=axes[1], color=colors[kappa])

        with df.HDF5File(mesh.mpi_comm(), fout, 'w') as out:
            out.write(mesh, 'mesh/')
            out.write(volumes, 'subdomains/')
            out.write(collagen, 'collagen_dist/')

        V = df.FunctionSpace(mesh, "CG", 1)
        theta = df.Function(V)
        theta.vector()[:] = collagen.array()[:]
    
        V2 = df.VectorFunctionSpace(mesh, "CG", 2)
    
        e1 = df.as_vector([1.0, 0.0])
        R = df.as_matrix(
            (
                (df.cos(theta), -df.sin(theta)),
                (df.sin(theta), df.cos(theta)),
            )
        )
    
        df.File(f"meshes_revisions/2d_mesh_{remove_type}_N_{N}_k_{kappa}_seed_{seed}.pvd") << df.project(R*e1, V2)
        

        write_collagen_to_file(fout)

axes[0].set_title("Endomysial collagen")
axes[1].set_title("Perimysial collagen")

axes[0].set_xlabel("Angle (-)")
axes[1].set_xlabel("Angle (-)")

axes[0].legend()
axes[1].legend()

plt.savefig("collagen_distribution.pdf", dpi=300)

plt.show()
