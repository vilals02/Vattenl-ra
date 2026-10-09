"""
Shallow-water equations — flow past a cylinder   (DOLFINx / FEniCSx)
=====================================================================

Mesh workflow (run once before this script):
    python convert_gmsh_xdmf.py cylinder.geo cylinder

PDE:   ∂_t U + ∇·F(U) = 0,    U = (h, hu, hv)ᵀ

Boundary conditions:
    left  (x=0)   inflow  — prescribe (h, hu, hv) = (h₀, h₀u₀, 0)
    right (x=L)   outflow — unconstrained (no boundary integral)
    top/bottom    slip    — hv = 0
    cylinder      slip    — (hu, hv)·n̂ = 0  (normal projection)

Time integration: explicit SSP-RK3 (Shu–Osher).
Output: swe_cylinder.xdmf + .h5  (open in ParaView).
"""

from mpi4py import MPI
import numpy as np
from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem import form
from ufl import inner, dx, TestFunction
from petsc4py import PETSc
import funcs as funky
import matplotlib.pyplot as plt


# ── parameters ────────────────────────────────────────────────────────────────
g      = 9.81
EPS    = 2e-3    # constant artificial viscosity
CFL    = 0.4
T_END  = 0.6
N_SNAP = 60

# must match cylinder.geo
L, H        = 10.0, 8.0
cx, cy, cr  = 3.0, 4.0, 1.0

gamma = 1.4
rho0, u0, v0, p0,  = 1.0, 3.0, 0, 6.2857

# ── load mesh from XDMF ───────────────────────────────────────────────────────
print("Loading mesh...", flush=True)
with io.XDMFFile(MPI.COMM_WORLD, "cylinder.xdmf", "r") as xf:
    msh = xf.read_mesh(name="Grid")   # change "Grid" if your convert script uses a different name

dim  = msh.topology.dim
fdim = dim - 1

# ── function space ────────────────────────────────────────────────────────────
V   = fem.functionspace(msh, ("Lagrange", 1, (4,)))   # U = (rho, u, v, P)
U_h = fem.Function(V, name="U")

Q0      = fem.functionspace(msh, ("DG", 0))
eps_f   = fem.Function(Q0, name="eps")
cells_Q = Q0.dofmap.list[:, 0]      # one dof per cell

# ── initial condition: uniform inflow ─────────────────────────────────────────
x   = V.tabulate_dof_coordinates()       # (n_dofs, 4)
cells_V = V.dofmap.list # (cells, 4)
arr = U_h.x.array.reshape(-1, 4)
arr[:, 0] = rho0
arr[:, 1] = u0
arr[:, 2] = v0
arr[:, 3] = p0
U_h.x.scatter_forward()

# ── boundary DOFs (coordinate-based) ─────────────────────────────────────────
x_dof = V.tabulate_dof_coordinates()    # (n_nodes, 4)
dist  = np.sqrt((x_dof[:, 0] - cx)**2 + (x_dof[:, 1] - cy)**2)

# left inflow: all three components prescribed
inflow_idx = np.where(np.isclose(x_dof[:, 0], 0.0))[0]

# top/bottom walls: hv = 0
dofs_hv = np.where(
    np.isclose(x_dof[:, 1], 0.0) | np.isclose(x_dof[:, 1], H)
)[0]

# cylinder: slip — project out the outward-normal momentum component
# tolerance = half the finest cell size near the cylinder (res_cyl = 0.02)
cyl_idx = np.where(np.abs(dist - cr) < 0.01)[0]
n_cyl   = x_dof[cyl_idx, :2] - np.array([cx, cy])
n_cyl  /= np.linalg.norm(n_cyl, axis=1, keepdims=True)

print(f"Boundary nodes — inflow: {len(inflow_idx)},  "
      f"top/bottom: {len(dofs_hv)},  cylinder: {len(cyl_idx)}", flush=True)
if len(cyl_idx) == 0:
    raise RuntimeError("No cylinder nodes found — check cx, cy, cr and the mesh.")

# ── helper functions ──────────────────────────────────────────────────────────
def compute_rhs():
    R = assemble_vector(R_form)
    R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return R.array.copy() * m_inv

def dt_cfl():
    """CFL time step: global min cell size / global max wave speed.
    Uses all nodal values — no cell indexing needed."""
    U    = U_h.x.array.reshape(-1, 4)
    h_c  = np.maximum(U[:, 0], 1e-4)   # floor avoids h→0 blowing up velocity
    u_c  = U[:, 1] / h_c
    v_c  = U[:, 2] / h_c
    lam  = np.abs(u_c) + np.abs(v_c) + np.sqrt(g * h_c)
    return float(CFL * h_K.min() / (lam.max() + 1e-30))

def ssp_rk3(dt):
    """Shu–Osher SSP-RK3; all BCs enforced after each stage."""
    u0 = U_h.x.array.copy()

    U_h.x.array[:] = u0 + dt * compute_rhs()
    apply_bcs(U_h.x.array);  U_h.x.scatter_forward()

    U_h.x.array[:] = 0.75*u0 + 0.25*(U_h.x.array + dt * compute_rhs())
    apply_bcs(U_h.x.array);  U_h.x.scatter_forward()

    U_h.x.array[:] = u0/3 + (2/3)*(U_h.x.array + dt * compute_rhs())
    apply_bcs(U_h.x.array);  U_h.x.scatter_forward()

def primitive():
    U = U_h.x.array.reshape(-1, 4)
    r = np.maximum(U[:, 0], 1e-10) # Avoid div by zero
    u, v = U[:, 1] / r, U[:, 2] / r
    p = np.maximum((gamma - 1) * (U[:, 3] - 0.5 * r * (u**2 + v**2)), 1e-10)
    return r, u, v, p          # numpy arrays, one value per node

def update_eps():
    rho, u, v, p = primitive()
    eps_K   = funky.compute_eps_K(p, rho, u, v, C, x, cells_V)
    eps_f.x.array[cells_Q] = eps_K
    eps_f.x.scatter_forward()

def apply_bcs(arr):
    U = arr.reshape(-1, 4)
    # inflow
    U[inflow_idx, 0] = rho0
    U[inflow_idx, 1] = u0
    U[inflow_idx, 2] = v0
    U[inflow_idx, 3] = p0
    # top/bottom slip
    U[dofs_hv, 2] = 0.0
    # cylinder slip: (hu,hv) -= ((hu,hv)·n̂) n̂
    u  = U[cyl_idx, 1].copy();  v = U[cyl_idx, 2].copy()
    dot = u * n_cyl[:, 0] + v * n_cyl[:, 1]
    U[cyl_idx, 1] -= dot * n_cyl[:, 0]
    U[cyl_idx, 2] -= dot * n_cyl[:, 1]

apply_bcs(U_h.x.array)

# ── cell sizes h_K = √(area_K) ────────────────────────────────────────────────
msh.topology.create_connectivity(dim, 0)
c2v   = msh.topology.connectivity(dim, 0)
ncel  = msh.topology.index_map(dim).size_local
geo   = msh.geometry.x
verts = np.array([c2v.links(k) for k in range(ncel)])
c     = geo[verts]
ab, ac = c[:, 1, :2] - c[:, 0, :2], c[:, 2, :2] - c[:, 0, :2]
h_K   = np.sqrt(0.5 * np.abs(ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]))

print(f"h_K min = {h_K.min():.4f},  EPS = {EPS:.4f}", flush=True)

# ── lumped mass vector (assembled once) ───────────────────────────────────────
one   = fem.Constant(msh, PETSc.ScalarType(np.ones(4)))
M_L   = assemble_vector(form(inner(one, TestFunction(V)) * dx))
M_L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
m_inv = 1.0 / M_L.array

# ── UFL residual (compiled once before the time loop) ────────────────────────
C = 0.5 #parameter
# Up = funky.conservative_to_primitive(U_h.x.array.reshape(-1, 4))
psi   = TestFunction(V)

gamma = 1.4
rho_, rhou_, rhov_, E_ = U_h[0], U_h[1], U_h[2], U_h[3]
u_ = rhou_ / rho_
v_ = rhov_ / rho_
p_ = (gamma - 1) * (E_ - 0.5 * rho_ * (u_**2 + v_**2))
update_eps()

F1    = [rhou_, rho_*np.square(u_)+p_, rhou_*v_, (E_+p_)*u_]
F2    = [rhov_, rhou_*v_, rho_*np.square(v_)+p_, (E_+p_)*v_]
div_F = sum((F1[i].dx(0) + F2[i].dx(1)) * psi[i] for i in range(4))
visc  = eps_f * sum((U_h[i].dx(0)*psi[i].dx(0) +
                   U_h[i].dx(1)*psi[i].dx(1)) for i in range(4))
R_form = form((-div_F - visc) * dx)

# ── warm up: force JIT compilation before the time loop ──────────────────────
print("Compiling form (first assembly)...", flush=True)
_ = compute_rhs()
print(f"Done.  Initial dt = {dt_cfl():.3e}", flush=True)

# ── time loop ─────────────────────────────────────────────────────────────────
vtk = io.VTKFile(msh.comm, "results/swe_cylinder.pvd", "w")

t             = 0.0
snap_interval = T_END / N_SNAP
next_snap     = snap_interval

eps   = 1e-3
mask  = np.abs(x[:, 1] - 0.1) < eps
order = np.argsort(x[mask, 0])
xs    = x[mask, 0][order]

snap_count = 0

xgrid = np.linspace(0, 1, np.shape(U_h.x.array.reshape(-1, 4))[0])

r, u, v, p = primitive()
plt.xlabel("x")
plt.ylabel("u")
plt.plot(xs, u[mask][order], "-o", ms=1, label=f"t = {t:.3f}")

vtk.write_function(U_h, t)
print(f"{'step':>6}  {'t':>8}  {'dt':>10}  {'min h':>10}")
step = 0
while t < T_END - 1e-12:
    dt = min(dt_cfl(), T_END - t, next_snap - t + 1e-14)
    update_eps()
    ssp_rk3(dt)
    t    += dt
    step += 1
    if t >= next_snap - 1e-12:
        snap_count += 1
        vtk.write_function(U_h, t)
        h_arr = U_h.x.array.reshape(-1, 4)[:, 0]
        print(f"{step:6d}  {t:8.4f}  {dt:10.2e}  {h_arr.min():10.4f}")
        next_snap += snap_interval
        if snap_count % 4 == 0:
                r, u, v, p = primitive()
                plt.xlabel("x")
                plt.ylabel("u")
                plt.plot(xs, u[mask][order], "-o", ms=1, label=f"t = {t:.3f}")

vtk.close()
print("Done — open swe_dambreak.pvd in ParaView.")
plt.legend()
plt.title(r"$u(x, y)$ at $y = 0.1$")
plt.savefig("BigKennyBoiExplosionCylinderEdition")
