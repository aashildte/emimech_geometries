"""

This script is based on the demo script here:
    https://github.com/MiroK/gemi/blob/master/demo/demo_single_mesh2d.py
adapted with some other parameters and project-specific geometries.

It is similar to generate_mesh.py but removes all extracellular space between the cells.

"""

# Show making mesh of given characteristic size for the sheet geometry
from gemi.sheet import sheet_geometry
from gemi.cells import make_rectangle
from functools import partial
import gmsh


# Specialize the cell by fixing its size
make_cell = partial(make_rectangle, dx=(102.0, 14.0))
clmax = '2.5'

# Silent gmsh and set the mesh size
gmsh.initialize(['', '-v', '0', '-clmax', clmax, '-anisoMin', '1'])

model = gmsh.model
gmsh.option.setNumber("Mesh.SmoothRatio", 3)
gmsh.option.setNumber("Mesh.AnisoMax", 1)
gmsh.option.setNumber("Mesh.Algorithm", 7)

# In our geoemtry we want to create sheet with 2 x 4 cells ...
ncells = (6, 12)
# ... that will have the following padding
pads = (5.0, 5.0)

# We get back a lookup table for checing how interfaces are connected to
# cells and extracellular space.
# NOTE: here we set shifts to None so that the cells will be glued to gether.
# Shifts which is larger then the bounding box of the cell allows as to create
# gaps between cells in the sheet
shifts = None #(104.0, 23)
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

from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
# Now we call on to `gmshnics` to get to mesh and mesh functions;
# NOTE: we are meshing a 2-dim problem
nodes, topologies = msh_gmsh_model(model, 2)
mesh, entity_fs = mesh_from_gmsh(nodes, topologies)

entity_fs[2].array()[:] -= 1

# At this point we are done with gmsh
gmsh.finalize()

import dolfin as df
import json

df.set_log_level(100)

facet_dim = mesh.topology().dim() - 1
# How many "facet cells" are there on the interface
membrane_cells = sum(sum(1 for f in df.SubsetIterator(entity_fs[facet_dim], color))
                     for color in all_interfaces)
print(f'Number of membrane facet {membrane_cells} / Total number of facets {mesh.num_entities(facet_dim)}')

from gemi.utils import mark_interfaces

simple_interface, etag, itag = mark_interfaces(entity_fs[facet_dim], connectivity)

# Just show of dumping to HDF5 ...
with df.HDF5File(mesh.mpi_comm(), f'meshes/2d_mesh_woe_{m}_{n}_{clmax}.h5', 'w') as out:
    out.write(mesh, 'mesh/')
    out.write(entity_fs[2], 'subdomains/')
    out.write(entity_fs[1], 'interfaces/')        

# And Paraview for visual inspection
df.File(f'meshes/2d_mesh_woe_{m}_{n}_{clmax}_subdomains.pvd') << entity_fs[2]
df.File(f'meshes/2d_mesh_woe_{m}_{n}_{clmax}_interfaces.pvd') << entity_fs[1]
df.File(f'meshes/2d_mesh_woe_{m}_{n}_{clmax}_interfaces.pvd') << simple_interface

