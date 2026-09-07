import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from scipy.sparse.linalg import spsolve, inv as sparse_inv
import matplotlib.pyplot as plt

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

#make mesh centered around origo

msh = mesh.create_rectangle(
    MPI.COMM_SELF,
    points=[[-0.5, -0.5], [0.5, 0.5]],
    n=[30, 30],
    cell_type=mesh.CellType.triangle,
)

p, e, t = dolfinx_to_pet(msh)

#plottten = plot_mesh(p,e,t)

#plt.show()

M = mass_assembler_2d(p, t)
b = load_assembler_2d(p, t, lambda x, y: 1.0)
A = stiffness_assembler_2d(p, t)

bx = np.ones(p.shape[1])
by = 2.0 * np.ones(p.shape[1])
C = convection_assembler_2d(p, t, bx, by)

import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

one = np.ones(p.shape[1])

r0 = 0.25
x0 = 0.3
y0 = 0
h = 0.001

exact_x = p[0]

M_inv = sparse_inv(M)
U0 = f.initial_profile(p, x0, y0, r0, len(p[1]))

# build the triangulation once, from your mesh connectivity
triangles = t.T if t.shape[0] == 3 else t
triangles = triangles.astype(int)
tri = mtri.Triangulation(p[0], p[1], triangles)

n_iterations = 1000
U_n = U0
for n in range(n_iterations):
    U_n = f.forward_euler_step(M_inv, C, U_n, h)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(tri, U_n, cmap="viridis", edgecolor="none")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
ax.set_title(f"Solution after {n_iterations} steps, h={h}")
fig.colorbar(surf, shrink=0.6)
plt.show()



print("1^T M 1           =", one @ (M @ one))
print("sum(b), f=1       =", np.sum(b))
print("||M1-b||_inf      =", np.linalg.norm(M @ one - b, np.inf))
print("||A1||_inf        =", np.linalg.norm(A @ one, np.inf))
print("||C1||_inf        =", np.linalg.norm(C @ one, np.inf))

print("M:", M.shape, "nnz =", M.nnz)
print("A:", A.shape, "nnz =", A.nnz)
print("C:", C.shape, "nnz =", C.nnz)

print("Point", p.shape)
