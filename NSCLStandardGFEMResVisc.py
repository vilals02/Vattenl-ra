import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType
from scipy.sparse.linalg import spsolve, inv as sparse_inv
import scipy as sp
import matplotlib.pyplot as plt

from mesh import dolfinx_to_pet, xdmf_to_pet
from fem_assemblers import (
    mass_assembler_2d,
    load_assembler_2d,
    stiffness_assembler_2d_NL_RV,
    convection_assembler_2d_NL,
    create_eps_K_array_NL_RV
)

import funcs as f

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

#make mesh centered around origo
# ── Read mesh ──────────────────────────────────────────────────────────────────

nx, ny = 40, 40  # divisions in x, y — bump up to resolve the spiral shock

Lx, Ly = 4.0, 4.0
hx, hy = Lx / nx, Ly / ny
h = max(hx, hy) 

msh = create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-2.0, -2.5), (2.0, 1.5)),
    n=(nx, ny),
    cell_type=CellType.triangle,
)
p, e, t = dolfinx_to_pet(msh)
x, y    = p[0], p[1]
n_dofs  = p.shape[1]

print(f"Nodes     : {n_dofs}")
print(f"Triangles : {t.shape[1]}")
print(f"Bnd edges : {e.shape[1]}")


M = mass_assembler_2d(p, t)
m = np.asarray(M.sum(axis=1)).ravel() 
M = sp.sparse.diags(m) 
b = load_assembler_2d(p, t, lambda x, y: 1.0)

print("M shape: ", M.shape)
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

# One-pass row elimination (K is already non-symmetric, so no need to
# zero the column or shift the RHS).
#   row i -> 0 ... 0  1  0 ... 0
#   b[i]  -> g_i

one = np.ones(p.shape[1])

max_time = 1 # seconds
r0 = 0.25
x0 = 0.3
y0 = 0
CFL = 0.05
k = CFL * h # timestep for finest mesh, h = 0.05
n_iterations = int(np.ceil(max_time / k))

exact_x = p[0]

M_inv = sparse_inv(M)

# build the triangulation once, from your mesh connectivity
triangles = t.T if t.shape[0] == 3 else t
triangles = triangles.astype(int)
tri = mtri.Triangulation(p[0], p[1], triangles)
minU = np.zeros(n_iterations)
maxU = np.zeros(n_iterations)
total_mass = np.zeros(n_iterations)
U_n = f.initial_profile_NL(p, p.shape[1])
U_np = U_n.copy()
for n in range(n_iterations):
    f1p, f2p = f.fp(U_n)
    S = stiffness_assembler_2d_NL_RV(p, t, U_n, U_np, k)
    C = convection_assembler_2d_NL(p, t, f1p, f2p)

    A = S + C
    U_np = U_n.copy()
    U_n = f.rk3_SSP_step(M_inv, A, U_n, k) 
    minU[n], maxU[n] = np.min(U_n), np.max(U_n)
    total_mass[n] = m @ U_n

rel_mass_err = np.abs(total_mass -  total_mass[0])/np.abs(total_mass[0])
errors = np.zeros(len(p[1]))

elapsed_time = k*n_iterations

eps_field = create_eps_K_array_NL_RV(p, t, U_n, U_np, k)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(tri, U_n, cmap="viridis", edgecolor="none")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
ax.set_title(rf"Solution after {max_time} seconds, k={k} for CFL: {CFL}")
fig.colorbar(surf, shrink=0.6)


fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(np.linspace(0, 1, n_iterations), rel_mass_err, label="total mass")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("total mass")
ax1.set_title(rf"Relative mass error as a function of time for CFL: {CFL}")


fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(np.linspace(0, 1, n_iterations), minU, label=r"$U_{min}$")
ax2.plot(np.linspace(0, 1, n_iterations), maxU, label=r"$U_{max}$")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("total mass")
ax2.set_title(rf"Range of solution $U_h$ at time $t$ for CFL: {CFL}")

fig3, ax3 = plt.subplots(figsize=(8, 6))
pc = ax3.tripcolor(tri, facecolors=eps_field, cmap="viridis")
ax3.set_aspect("equal")
fig3.colorbar(pc)

plt.show()


print("1^T M 1           =", one @ (M @ one))
print("sum(b), f=1       =", np.sum(b))
print("||M1-b||_inf      =", np.linalg.norm(M @ one - b, np.inf))
print("||C1||_inf        =", np.linalg.norm(C @ one, np.inf))

print("M:", M.shape, "nnz =", M.nnz)
print("C:", C.shape, "nnz =", C.nnz)
print("triangles: ", t.shape)
print("First triangle: ", t[:,0])
print("Max index: ", np.max(t.reshape(-1)))
print("Edges: ", e.shape)

print("Point", p.shape)
