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
# Usage: python run_convection_cfl.py [mesh_file.xdmf]
mesh_file = sys.argv[1] if len(sys.argv) > 1 else "unit_circle.xdmf"

# toggles
euler = False
integration = 'Euler' if euler else 'RK4'

OUT_DIR = "plots_for_1-3"
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

beta = np.zeros(p.shape)
for node in range(n_dofs):
    beta[0, node], beta[1, node] = f.get_speed(p[0, node], p[1, node])

# ── Time stepping ──────────────────────────────────────────────────────────────
max_time = 0.2  # seconds
r0 = 0.25
x0 = 0.3
y0 = 0
CFL = 36
h = 0.08  # hardcoded: the same as the one in unit_circle.geo
# k = CFL * h / ||beta||_{L-inf(Omega)}, where ||beta||_{L-inf} = max_x |beta(x)|
# (largest Euclidean speed over the nodes)
max_speed = np.max(np.linalg.norm(beta, axis=0))
k = CFL * h / max_speed
n_iterations = int(np.ceil(max_time / k))
elapsed_time = k * n_iterations

print(f"h = {h}, max|beta| = {max_speed:.3f}, CFL = {CFL}, k = {k:.3e}, "
      f"steps = {n_iterations}, elapsed = {elapsed_time:.4f}")
if k > max_time:
    print(f"WARNING: k = {k:.3f} > max_time = {max_time}: the run is a single "
          f"step to t = {elapsed_time:.3f}")

M_inv = sparse_inv(M)
U_n = f.initial_profile_A(p, x0, y0, r0, n_dofs)

total_mass = np.zeros(n_iterations)
for n in range(n_iterations):
    if euler:
        U_n = f.forward_euler_step(M_inv, C, U_n, k)
    else:
        U_n = f.rk4_step(M_inv, C, U_n, k)
    total_mass[n] = np.sum(U_n)
total_mass = total_mass / total_mass[0]

# ── Error ──────────────────────────────────────────────────────────────────────
# Exact solution at the time actually reached (elapsed_time): pull the point
# back along the flow (rotate by -theta)
theta = 2 * np.pi * elapsed_time

errors = np.zeros(n_dofs)
for n in range(n_dofs):
    x1 = p[0, n] * np.cos(theta) + p[1, n] * np.sin(theta)
    x2 = -p[0, n] * np.sin(theta) + p[1, n] * np.cos(theta)
    errors[n] = f.exact_solution_A(x1, x2, x0, y0, r0) - U_n[n]

eh = np.sqrt(errors @ (M @ errors))
print(f"Error: {eh}")

# ── Append result to CSV (header written only if the file is new) ─────────────
write_header = not os.path.exists(CSV_FILE)
with open(CSV_FILE, "a", newline="") as fh:
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(["mesh_file", "n_nodes", "n_triangles", "integration",
                         "CFL", "h", "k", "n_steps", "max_time",
                         "T", "L2_error"])
    writer.writerow([os.path.basename(mesh_file), n_dofs, t.shape[1],
                     integration, CFL, f"{h:.6e}",
                     f"{k:.6e}", n_iterations, max_time,
                     f"{elapsed_time:.6f}", f"{eh:.6e}"])

# ── Plots ──────────────────────────────────────────────────────────────────────
tag = f"{n_dofs}_{integration}_CFL{CFL}"

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_trisurf(tri, U_n, cmap="viridis", edgecolor="none")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
ax.set_title(f"Solution after {elapsed_time:.3f} seconds, k={k:.2e}")
fig.colorbar(surf, shrink=0.6)
fig.savefig(os.path.join(OUT_DIR, f"sol-plot_{tag}"))
plt.close(fig)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(np.linspace(0, elapsed_time, n_iterations), total_mass, label="total mass")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("total mass")
ax1.set_title("Total mass as a function of time")
fig1.savefig(os.path.join(OUT_DIR, f"mass-plot_{tag}"))
plt.close(fig1)

# ── Diagnostics ────────────────────────────────────────────────────────────────
print("1^T M 1           =", one @ (M @ one))
print("sum(b), f=1       =", np.sum(b))
print("||M1-b||_inf      =", np.linalg.norm(M @ one - b, np.inf))
print("||C1||_inf        =", np.linalg.norm(C @ one, np.inf))
print("M:", M.shape, "nnz =", M.nnz)
print("C:", C.shape, "nnz =", C.nnz)