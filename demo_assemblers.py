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


msh = mesh.create_unit_square(
    MPI.COMM_SELF, 4, 4,
    cell_type=mesh.CellType.triangle
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

one = np.ones(p.shape[1])

max_time = 1
n_timesteps = 100
dt = max_time / n_timesteps

N = p.shape[1]
path_matrix_x = np.zeros((N, n_timesteps + 1))
path_matrix_y = np.zeros((N, n_timesteps + 1))

# work on a copy so the original mesh points p are untouched
p_track = p.copy()
path_matrix_x[:, 0] = p_track[0]
path_matrix_y[:, 0] = p_track[1]

r0 = 0.25
x0 = 0.3
y0 = 0

U0 = f.initial_profile(p, x0, y0, r0, len(p[1]))

for j in range(n_timesteps):
    for i in range(N):
        x = p_track[0, i]
        y = p_track[1, i]

        vx, vy = f.get_speed(x, y)

        nx = x + dt * vx
        ny = y + dt * vy

        p_track[0, i] = nx
        p_track[1, i] = ny

        path_matrix_x[i, j + 1] = nx
        path_matrix_y[i, j + 1] = ny

# --- plot the particle paths ---
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

# color each path by its initial U0 value, for visual reference
norm = mcolors.Normalize(vmin=U0.min(), vmax=U0.max())
cmap = cm.viridis

for i in range(N):
    z_i = np.full(n_timesteps + 1, U0[i])  # constant height = U0[i]
    ax.plot(
        path_matrix_x[i, :],
        path_matrix_y[i, :],
        z_i,
        color=cmap(norm(U0[i])),
        linewidth=0.8,
    )

# mark starting points
ax.scatter(
    path_matrix_x[:, 0],
    path_matrix_y[:, 0],
    U0,
    c=U0,
    cmap=cmap,
    s=15,
    zorder=5,
    label="start",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U0[i]")
ax.set_title(f"Particle paths lifted to initial U0 value ({n_timesteps} steps, dt={dt:.4f})")

mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(U0)
fig.colorbar(mappable, ax=ax, shrink=0.6, label="U0")

plt.show()







# r0 = 0.25
# x0 = 0.3
# y0 = 0
# h = 0.001


# exact_x = p[0]

# M_inv = sparse_inv(M)
# U0 = f.initial_profile(p, x0, y0, r0, len(p[1]))

# n_iterations = 100
# U_n = U0
# for n in range(n_iterations):
#     U_n = f.forward_euler_step(M_inv, C, U_n, h)
    
#     plt.plot(U_n)

# plt.show()



print("1^T M 1           =", one @ (M @ one))
print("sum(b), f=1       =", np.sum(b))
print("||M1-b||_inf      =", np.linalg.norm(M @ one - b, np.inf))
print("||A1||_inf        =", np.linalg.norm(A @ one, np.inf))
print("||C1||_inf        =", np.linalg.norm(C @ one, np.inf))

print("M:", M.shape, "nnz =", M.nnz)
print("A:", A.shape, "nnz =", A.nnz)
print("C:", C.shape, "nnz =", C.nnz)

print("Point", p.shape)
