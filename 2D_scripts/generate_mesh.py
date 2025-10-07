"""

This script is based on the demo script here:
    https://github.com/MiroK/gemi/blob/master/demo/demo_single_mesh2d.py
adapted with some other parameters and project-specific geometries.

"""
from gemi.sheet import sheet_geometry
from gemi.cells import make_plus2d
from gemi.utils import mark_interfaces
from functools import partial
import gmsh
from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
import dolfin as df
import json

clmax = '1'

# With specific parameters for cardiomyocyte geometries
make_cell = partial(make_plus2d, dx0=(2.0, 12.0), dx1=(100.0, 4.0))
gmsh.initialize(['', '-v', '0', '-clmax', clmax, '-anisoMin', '1'])
model = gmsh.model

# These parameters make the mesh smoother
gmsh.option.setNumber("Mesh.SmoothRatio", 3)
gmsh.option.setNumber("Mesh.AnisoMax", 1)
gmsh.option.setNumber("Mesh.Algorithm", 7)

# In our geoemtry we want to create sheet with m x n cells ...
m = 8
n = 16
ncells = (m, n)
# ... that will have the following padding
pads = (10.0, 5.0)

pad = 3
# gaps between cells in the sheet
shifts = (104.0, 23 + pad)

model, connectivity = sheet_geometry(model, make_cell=make_cell, ncells=ncells, pads=pads,
                                     shifts=shifts)

# From connectivity we can ask about the nature of interfaces
# Cell taged 1 is extracellular
all_interfaces = [facet for facet in connectivity if len(connectivity[facet]) > 1]
# Which of these are interfaces between EMI cells - here one would have gap
# junctions
gj_interfaces = [facet for facet in all_interfaces if 1 not in connectivity[facet]]

for facet, cells in connectivity.items():
    print(f'Facet {facet} is connected to cells {cells}')

model.occ.synchronize()

# We can checkout the geometry in gmsh
if True:
    gmsh.fltk.initialize()
    gmsh.fltk.run()

nodes, topologies = msh_gmsh_model(model, 2)
mesh, entity_fs = mesh_from_gmsh(nodes, topologies)

# NOTE this makes the extracellular space 0, which is just to be consisten with the emimechanics code assumptions
entity_fs[2].array()[:] -= 1

# We can dump the mesh for gmsh here (and continue with e.g. meshio)
#gmsh.write('meshes/demo2d.msh')

# At this point we are done with gmsh
gmsh.finalize()
df.set_log_level(100)

facet_dim = mesh.topology().dim() - 1
# How many "facet cells" are there on the interface
membrane_cells = sum(sum(1 for f in df.SubsetIterator(entity_fs[facet_dim], color))
                     for color in all_interfaces)
print(f'Number of membrane facet {membrane_cells} / Total number of facets {mesh.num_entities(facet_dim)}')

simple_interface, etag, itag = mark_interfaces(entity_fs[facet_dim], connectivity)

# Just show of dumping to HDF5 ...
with df.HDF5File(mesh.mpi_comm(), f'meshes_refinement_pad_10/2d_mesh_{m}_{n}_{clmax}.h5', 'w') as out:
    out.write(mesh, 'mesh/')
    out.write(entity_fs[2], 'subdomains/')
    out.write(entity_fs[1], 'interfaces/')        

#with open('demo2d.json', 'w') as out:
#    json.dump(connectivity, out)

# And Paraview for visual inspection
df.File(f'meshes/2d_mesh_{m}_{n}_{clmax}_subdomains.pvd') << entity_fs[2]
df.File(f'meshes/2d_mesh_{m}_{n}_{clmax}_interfaces.pvd') << entity_fs[1]
#df.File(f'meshes/2d_mesh_{m}_{n}_{clmax}_interfaces.pvd') << simple_interface
