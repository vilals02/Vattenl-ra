import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from scipy.sparse.linalg import spsolve, inv as sparse_inv
import matplotlib.pyplot as plt

from mesh import dolfinx_to_pet, xdmf_to_pet
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
# ── Read mesh ──────────────────────────────────────────────────────────────────

p, e, t = xdmf_to_pet("unit_circle.xdmf")
x, y    = p[0], p[1]
n_dofs  = p.shape[1]

print(f"Nodes     : {n_dofs}")
print(f"Triangles : {t.shape[1]}")
print(f"Bnd edges : {e.shape[1]}")

#plottten = plot_mesh(p,e,t)

#plt.show()

M = mass_assembler_2d(p, t)
b = load_assembler_2d(p, t, lambda x, y: 1.0)

bx = np.ones(p.shape[1])
by = 2.0 * np.ones(p.shape[1])
C = convection_assembler_2d(p, t, bx, by)

print("M shape: ", M.shape)
print("C shape: ", C.shape)
print("Num points shape: ", p.shape)

# ── Boundary conditions ────────────────────────────────────────────────────────
#
# All boundary nodes from e; split by sign of x:
#   x ≤ 0  →  inflow half   →  u = 1
#   x > 0  →  outflow half  →  u = 0

all_bnd = np.unique(e.reshape(-1))
left_mask  = x[all_bnd] <= 0.0
left_nodes  = all_bnd[ left_mask]
right_nodes = all_bnd[~left_mask]

dirichlet_nodes  = np.concatenate([left_nodes,  right_nodes])
dirichlet_values = np.concatenate([np.ones(len(left_nodes)),
                                   np.zeros(len(right_nodes))])

# One-pass row elimination (K is already non-symmetric, so no need to
# zero the column or shift the RHS).
#   row i -> 0 ... 0  1  0 ... 0
#   b[i]  -> g_i
for i, g in zip(dirichlet_nodes, dirichlet_values):
    C[i, :] = 0.0
    C[i, i] = 1.0
    b[i]    = g


import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

one = np.ones(p.shape[1])

max_time = 0.2 # seconds
r0 = 0.25
x0 = 0.3
y0 = 0
k = 0.00025 # timestep for finest mesh, h = 0.05
n_iterations = int(np.ceil(max_time / k))

exact_x = p[0]

M_inv = sparse_inv(M)
U0 = f.initial_profile_A(p, x0, y0, r0, len(p[1]))

# build the triangulation once, from your mesh connectivity
triangles = t.T if t.shape[0] == 3 else t
triangles = triangles.astype(int)
tri = mtri.Triangulation(p[0], p[1], triangles)

U_n = U0
total_mass = np.zeros(n_iterations)
for n in range(n_iterations):
    U_n = f.rk4_step(M_inv, C, U_n, k)
    total_mass[n] = np.sum(U_n)


errors = np.zeros(len(p[1]))

elapsed_time = k*n_iterations

#Calculate error vector
for n in range(len(p[1])):

    x1_c = p[0, n]
    x2_c = p[1, n] 

    x10 = x1_c - elapsed_time*f.get_speed(x1_c, x2_c)[0]
    x20 = x2_c - elapsed_time*f.get_speed(x1_c, x2_c)[1]

    u = f.exact_solution(x10, x20,x0, y0, r0)

    errors[n] = u - U_n[n]  

#Compute norm of eh

eh = np.sqrt(errors.T@M@errors)

print(f"Error: {eh}")




fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(tri, U_n, cmap="viridis", edgecolor="none")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
ax.set_title(f"Solution after {max_time} seconds, k={k}")
fig.colorbar(surf, shrink=0.6)

fig.savefig(f"plots/sol-plot_stable")

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(np.linspace(0, 1, n_iterations), total_mass, label="total mass")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("total mass")
ax1.set_title("Total mass as a function of time")

fig1.savefig(f"plots/mass-plot_stable")



print("1^T M 1           =", one @ (M @ one))
print("sum(b), f=1       =", np.sum(b))
print("||M1-b||_inf      =", np.linalg.norm(M @ one - b, np.inf))
print("||C1||_inf        =", np.linalg.norm(C @ one, np.inf))

print("M:", M.shape, "nnz =", M.nnz)
print("C:", C.shape, "nnz =", C.nnz)

print("Point", p.shape)
