
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import dolfin as df
import matplotlib.pyplot as plt

def get_mesh_dimensions(mesh):

    coords = mesh.coordinates()[:]

    xmax = max(coords[:,0])
    xmin = min(coords[:,0])
    ymax = max(coords[:,1])
    ymin = min(coords[:,1])

    return xmin, xmax, ymin, ymax

def get_block_coords(mesh, N):
    xmin, xmax, ymin, ymax = get_mesh_dimensions(mesh)

    xcoord = xmin - N/2
    xcoords = []

    while xcoord < xmax + N:
        xcoords.append(xcoord)
        xcoord += N
    
    ycoord = ymin - N/2
    ycoords = []

    while ycoord < ymax + N:
        ycoords.append(ycoord)
        ycoord += N

    return np.array(xcoords), np.array(ycoords)


def categorize_cg_values(u_cg: df.Function, cell_markers: df.MeshFunction):
    """
    Collect CG-1 nodal values based on cell markers.

    Parameters
    ----------
    u_cg : df.Function
        A function in CG-1 space (continuous, nodal).
    cell_markers : df.MeshFunction('size_t')
        Cell-wise markers (e.g. 1 for endomysium, 2 for perimysium).

    Returns
    -------
    dict
        Dictionary with two keys: "endomysium" and "perimysium".
        Each maps to a list of CG-1 values. If a dof belongs to
        cells in multiple categories, its value appears in both lists.
    """
    V_cg = u_cg.function_space()
    mesh = V_cg.mesh()

    cg_values = u_cg.vector().get_local()
    dofmap = V_cg.dofmap()

    categories = {
        "endomysium": [],
        "perimysium": []
    }

    for cell in df.cells(mesh):
        marker = cell_markers[cell]
        dofs = dofmap.cell_dofs(cell.index())
        for dof in dofs:
            val = cg_values[dof]
            if marker == 1:
                categories["endomysium"].append(val)
            elif marker == 2:
                categories["perimysium"].append(val)

    categories["endomysium"] = np.array(categories["endomysium"])
    categories["perimysium"] = np.array(categories["perimysium"])

    return categories




def assign_collagen_distribution(mesh, volumes, mu=0, kappa_endomysium=0, kappa_perimysium=1, N=10, plot=False, axis_endo=None, axis_peri=None, color=None):
    """

    Defines collagen alignment to points in the mesh, in the following manner:
    * Divide the (2D) domain into squares of size NxN (µm)
    * Assign an angle to those from a von Mises distribution
    * Do an interpolation on all mesh point

    """
    print("here")
    U_DG = df.FunctionSpace(mesh, "DG", 0)
    xi_perimysium = df.Function(U_DG)

    perimysium_id = 2
    volume_values = np.array(volumes.array()[:])
    xi_perimysium.vector()[:] = np.where(volume_values==perimysium_id, 1, 0)
    
    xcoords, ycoords = get_block_coords(mesh, N)
    values_endo = np.random.vonmises(mu=mu, kappa=kappa_endomysium, size=(len(xcoords), len(ycoords)))
    values_peri = np.random.vonmises(mu=mu, kappa=kappa_perimysium, size=(len(xcoords), len(ycoords)))

    ip_endo = RegularGridInterpolator(
        (xcoords, ycoords),
        values_endo,
        bounds_error=False,
        fill_value=0,
        method="nearest",
    )
    
    ip_peri = RegularGridInterpolator(
        (xcoords, ycoords),
        values_peri,
        bounds_error=False,
        fill_value=0,
        method="nearest",
    )

    U = df.FunctionSpace(mesh, "CG", 1)
    
    v_d = U.dofmap().dofs()
    mesh_coords = U.tabulate_dof_coordinates()[:]
    mesh_xcoords = np.array(mesh_coords[:, 0])
    mesh_ycoords = np.array(mesh_coords[:, 1])

    ip_values_endo = ip_endo((mesh_xcoords, mesh_ycoords))
    ip_values_peri = ip_peri((mesh_xcoords, mesh_ycoords))

    dist_endo = df.Function(U, name="Collagen endomysial distribution")
    dist_endo.vector()[v_d] = ip_values_endo

    dist_peri = df.Function(U, name="Collagen perimysial distribution")
    dist_peri.vector()[v_d] = ip_values_peri

    dist = df.project(dist_endo*(1-xi_perimysium) + dist_peri*xi_perimysium, U)

    collagen_dist_mf = df.MeshFunction('double', mesh, 0)
    collagen_dist_mf.array()[:] = dist.vector()[:]
    
    if plot:

        categories = categorize_cg_values(dist_endo, volumes)
        end = categories["endomysium"]
        
        categories = categorize_cg_values(dist_peri, volumes)
        peri = categories["perimysium"]

        axis_endo.set_xlim(-3.14, 3.14)
        bins = 11

        if kappa_endomysium==0:
            alpha=0.6
        else:
            alpha=0.3
        
        edgecolor="black"
    
        axis_endo.hist(end, bins=bins, alpha=alpha, density=True, color=color, edgecolor=edgecolor, label=r"$\kappa =$ " + str(kappa_endomysium))
        axis_peri.hist(peri, bins=bins, alpha=alpha, density=True, color=color, edgecolor=edgecolor, label=r"$\kappa =$ " + str(kappa_perimysium)) 

        axis_endo.set_xticks([-3.14, -3.14/2, 0, 3.14/2, 3.14], [r"-$\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
        axis_peri.set_xticks([-3.14, -3.14/2, 0, 3.14/2, 3.14], [r"-$\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
    return collagen_dist_mf
    
