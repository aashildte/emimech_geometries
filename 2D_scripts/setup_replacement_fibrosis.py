"""

Åshild Telle / University of Washington / 2023

Supporting functions for the main script for generating fibrotic geometries.

"""

import numpy as np
import dolfin as df
from mpi4py import MPI
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import random

def find_middle_cells_transversely(mesh, volumes, num_cells_transverse, num_cells_to_remove, pad, seed=None):
    coords = mesh.coordinates()[:]

    cmax = max(coords[:,0])
    cmin = min(coords[:,0])

    dx = (cmax - cmin)/num_cells_transverse

    x_start = cmin + (num_cells_transverse // 2 - pad)*dx
    x_stop = cmin + (num_cells_transverse // 2 + pad)*dx
    ids = set()


    for subdomain_id, node_ids in zip(volumes.array()[:], mesh.cells()[:]):
        for node_id in node_ids:
            if x_start < coords[node_id][0] < x_stop:
                ids.add(subdomain_id)
                break

    if 0 in ids:
        ids.remove(0)
    
    # then remove cells 

    if seed is not None:
        random.seed(seed)

    remove_ids = random.sample(list(ids), k=num_cells_to_remove)
    remove_ids.sort()
    print(remove_ids)
    
    return remove_ids


def find_middle_cells_longitudinally(mesh, volumes, num_cells_longitudinal, num_cells_to_remove, pad, seed=None):
    coords = mesh.coordinates()[:]

    cmax = max(coords[:,1])
    cmin = min(coords[:,1])

    dx = (cmax - cmin)/num_cells_longitudinal

    x_start = cmin + (num_cells_longitudinal // 2 - pad)*dx
    x_stop = cmin + (num_cells_longitudinal // 2 + pad)*dx
    print(pad) 
    print(cmax, cmin, dx, x_start, x_stop)

    ids = set()
    
    for subdomain_id, node_ids in zip(volumes.array()[:], mesh.cells()[:]):
        for node_id in node_ids:
            if x_start < coords[node_id][1] < x_stop:
                ids.add(subdomain_id)
                break

    if 0 in ids:
        ids.remove(0)

    if seed is not None:
        random.seed(seed)

    remove_ids = random.sample(list(ids), k=num_cells_to_remove)
    remove_ids.sort()
    print(remove_ids)
    
    return remove_ids


def find_random_cells(mesh, volumes, num_cells, seed, min_cell_id, max_cell_id):
    if seed is not None:
        random.seed(seed)

    cells = [x for x in range(min_cell_id, max_cell_id+1)] # list(set(volumes.array()[:]))

    ids = random.sample(cells, k=num_cells)
    print(ids)
    
    return ids


def replace_cells_with_matrix(mesh, volumes, cell_idts, replacement_id):
    N = volumes.size()
    
    for i in range(N):
        n = volumes.array()[i]
        if n in cell_idts:
            volumes.array()[i] = replacement_id


def generate_fibrotic_tissue(mesh, volumes, cell_idts):
    # convention: 0 = fibrotic, 1 = healthy

    print(cell_idts)
    
    volumes_array = volumes.array()[:]
    new_volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    matrix_idt = 0

    volumes_array = np.where(volumes_array == 0, -1, volumes_array) # hack

    for cell_idt in cell_idts:
        new_array = np.where(volumes_array == cell_idt, 0, volumes_array)
        volumes_array = new_array

    new_volumes.array()[:] = np.where(volumes_array == 0, 0, 1)
    return new_volumes
