import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from scipy.sparse.linalg import spsolve, inv as sparse_inv
import matplotlib.pyplot as plt

import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mesh import dolfinx_to_pet, plot_mesh
from fem_assemblers import (
    mass_assembler_2d,
    load_assembler_2d,
    stiffness_assembler_2d,
    convection_assembler_2d,
)

import funcs as f

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# toggle initial condition
initial_A = True
init_cond = 'A' if initial_A else 'B'

#make mesh centered around origo

msh = mesh.create_rectangle(
    MPI.COMM_SELF,
    points=[[-1, -1], [1, 1]],
    n=[50, 50],
    cell_type=mesh.CellType.triangle,
)

p, e, t = dolfinx_to_pet(msh)

# build the triangulation once, from your mesh connectivity
triangles = t.T if t.shape[0] == 3 else t
triangles = triangles.astype(int)
tri = mtri.Triangulation(p[0], p[1], triangles)

M = mass_assembler_2d(p, t)
b = load_assembler_2d(p, t, lambda x, y: 1.0)
A = stiffness_assembler_2d(p, t)

bx = np.ones(p.shape[1])
by = 2.0 * np.ones(p.shape[1])
C = convection_assembler_2d(p, t, bx, by)

one = np.ones(p.shape[1])

time = 0.2


r0 = 0.25
x0 = 0.3
y0 = 0

X = np.zeros(p.shape)
U_n = np.zeros(p.shape[1])
for i in range(len(p[1])):
    x1 = p[0, i]*np.cos(2*np.pi*time) - p[1, i]*np.sin(2*np.pi*time)
    x2 = p[0, i]*np.sin(2*np.pi*time) + p[1, i]*np.cos(2*np.pi*time)
    X[0, i], X[1, i] = x1, x2
    if initial_A:
        U_n[i] = f.exact_solution_A(x1, x2, x0, y0, r0)
    else: 
        U_n[i] = f.exact_solution_B(x1, x2, x0, y0, r0)

new_tri = mtri.Triangulation(X[0], X[1], triangles)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(new_tri, U_n, cmap="viridis", edgecolor="none")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
ax.set_title(f"Solution after {time} seconds")
fig.colorbar(surf, shrink=0.6)

fig.savefig(f"plots_Exact/short_{init_cond}")