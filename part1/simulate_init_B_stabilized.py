import os
import sys
import csv

import numpy as np
from scipy.sparse.linalg import inv as sparse_inv
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mesh import xdmf_to_pet
from fem_assemblers import (
    mass_assembler_2d,
    load_assembler_2d,
    stiffness_assembler_2d,
    convection_assembler_2d,
)
import funcs as f

# ── Settings ───────────────────────────────────────────────────────────────────
# Usage: python run_convection_B_stabilized.py [mesh_file.xdmf]
mesh_file = sys.argv[1] if len(sys.argv) > 1 else "unit_circle.xdmf"

OUT_DIR = "plots_Bs"
CSV_FILE = os.path.join(OUT_DIR, "errors.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Read mesh ──────────────────────────────────────────────────────────────────
p, e, t = xdmf_to_pet(mesh_file)
x, y = p[0], p[1]
n_dofs = p.shape[1]

print(f"Mesh      : {mesh_file}")
print(f"Nodes     : {n_dofs}")
print(f"Triangles : {t.shape[1]}")
print(f"Bnd edges : {e.shape[1]}")

triangles = (t.T if t.shape[0] == 3 else t).astype(int)
tri = mtri.Triangulation(p[0], p[1], triangles)

# Mesh size h = longest edge in the mesh
pts = p[:2].T
edge_lengths = np.concatenate([
    np.linalg.norm(pts[triangles[:, a]] - pts[triangles[:, b]], axis=1)
    for a, b in [(0, 1), (1, 2), (2, 0)]
])

# ── Assemble ───────────────────────────────────────────────────────────────────
M = mass_assembler_2d(p, t)
b = load_assembler_2d(p, t, lambda x, y: 1.0)
S = stiffness_assembler_2d(p, t)

bx = -2 * np.pi * p[1, :]
by = 2 * np.pi * p[0, :]
C = convection_assembler_2d(p, t, bx, by)

A = S + C

print("M shape: ", M.shape)
print("C shape: ", C.shape)
print("A shape: ", A.shape)
print("Num points shape: ", p.shape)

# ── Boundary conditions ────────────────────────────────────────────────────────
all_bnd = np.unique(e.reshape(-1))
left_mask = x[all_bnd] <= 0.0
left_nodes = all_bnd[left_mask]
right_nodes = all_bnd[~left_mask]

dirichlet_nodes = np.concatenate([left_nodes, right_nodes])
dirichlet_values = np.concatenate([np.ones(len(left_nodes)),
                                   np.zeros(len(right_nodes))])

for node, g in zip(dirichlet_nodes, dirichlet_values):
    C[node, :] = 0.0
    C[node, node] = 1.0
    b[node] = g

one = np.ones(n_dofs)

# ── Time stepping ──────────────────────────────────────────────────────────────
max_time = 1  # seconds
r0 = 0.25
x0 = 0.3
y0 = 0
CFL = 0.9
h = 0.05
k = CFL * 0.1 * h**2
n_iterations = int(np.ceil(max_time / k))
elapsed_time = k * n_iterations

print(f"h = {h:.4f}, k = {k:.3e}, steps = {n_iterations}")

M_inv = sparse_inv(M)
U_n = f.initial_profile_B(p, x0, y0, r0, n_dofs)

total_mass = np.zeros(n_iterations)
for n in range(n_iterations):
    U_n = f.rk4_step(M_inv, A, U_n, k)  # Stabilized
    total_mass[n] = np.sum(U_n)
total_mass = total_mass / total_mass[0]

# ── Error ──────────────────────────────────────────────────────────────────────
# Exact solution at time T: pull the point back along the flow (rotate by -theta)
theta = 2 * np.pi * elapsed_time

errors = np.zeros(n_dofs)
for n in range(n_dofs):
    x1 = p[0, n] * np.cos(theta) + p[1, n] * np.sin(theta)
    x2 = -p[0, n] * np.sin(theta) + p[1, n] * np.cos(theta)
    errors[n] = f.exact_solution_B(x1, x2, x0, y0, r0) - U_n[n]

eh = np.sqrt(errors @ (M @ errors))
print(f"Error: {eh}")

# ── Append result to CSV (header written only if the file is new) ─────────────
write_header = not os.path.exists(CSV_FILE)
with open(CSV_FILE, "a", newline="") as fh:
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(["mesh_file", "n_nodes", "n_triangles", "h", "k",
                         "n_steps", "T", "L2_error"])
    writer.writerow([os.path.basename(mesh_file), n_dofs, t.shape[1],
                     f"{h:.6e}", f"{k:.6e}", n_iterations,
                     f"{elapsed_time:.6f}", f"{eh:.6e}"])

# ── Plots ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(tri, U_n, cmap="viridis", edgecolor="none")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
ax.set_title(f"Solution after {max_time} seconds, k={k:.2e}")
fig.colorbar(surf, shrink=0.6)
fig.savefig(os.path.join(OUT_DIR, f"sol-plot_{n_dofs}"))
plt.close(fig)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(np.linspace(0, max_time, n_iterations), total_mass, label="total mass")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("total mass")
ax1.set_title("Total mass as a function of time")
fig1.savefig(os.path.join(OUT_DIR, f"mass-plot_{n_dofs}"))
plt.close(fig1)

# ── Diagnostics ────────────────────────────────────────────────────────────────
print("1^T M 1           =", one @ (M @ one))
print("sum(b), f=1       =", np.sum(b))
print("||M1-b||_inf      =", np.linalg.norm(M @ one - b, np.inf))
print("||C1||_inf        =", np.linalg.norm(C @ one, np.inf))
print("M:", M.shape, "nnz =", M.nnz)
print("C:", C.shape, "nnz =", C.nnz)