"""
Shallow-water equations — 2-D dam break   (DOLFINx / FEniCSx)
==============================================================

PDE (conservation form):

    ∂_t U + ∇·F(U) = 0,    U = (h, hu, hv)ᵀ

Fluxes:
    F₁ = (hu,   hu²+½gh²,  huv    )
    F₂ = (hv,   huv,       hv²+½gh²)

Weak form (no IBP on convection):

    M_L ∂_t U = −∫_Ω (∇·F(U)) ψ dx  −  ε ∫_Ω ∇U:∇ψ dx

M_L: lumped (row-sum) mass matrix.  ε: constant artificial viscosity.
Time integration: explicit SSP-RK3 (Shu–Osher).
Output: swe_dambreak.pvd  (open in ParaView).
"""

from mpi4py import MPI
import numpy as np
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem import form
from ufl import inner, dx, TestFunction
from petsc4py import PETSc

# ── parameters ───────────────────────────────────────────────────────────────
g      = 9.81    # gravity
EPS    = 5e-3    # constant artificial viscosity (~0.5 * h_min * wave_speed)
CFL    = 0.4     # CFL number
T_END  = 0.5     # final time
N_SNAP = 20      # snapshots written to disk

# ── mesh ─────────────────────────────────────────────────────────────────────
nx, ny = 200, 40
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [1.0, 0.2]],
    [nx, ny],
    mesh.CellType.triangle,
)
dim  = msh.topology.dim
fdim = dim - 1

# ── function space ────────────────────────────────────────────────────────────
V   = fem.functionspace(msh, ("Lagrange", 1, (3,)))   # U = (h, hu, hv)
U_h = fem.Function(V, name="U")

# ── initial condition: dam break ──────────────────────────────────────────────
x   = V.tabulate_dof_coordinates()       # (n_dofs, 3)
arr = U_h.x.array.reshape(-1, 3)
L   = x[:, 0] < 0.5                     # left half: deep water
arr[L,  0] = 1.0                         # h = 1  (left)
arr[~L, 0] = 0.5                         # h = 0.5 (right)
# hu = hv = 0 everywhere initially
U_h.x.scatter_forward()

# ── slip-wall boundary DOFs ───────────────────────────────────────────────────
# hv = 0 on top/bottom walls  (no normal flux through y = 0, 0.2)
hw      = mesh.locate_entities_boundary(msh, fdim,
              lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 0.2))
dofs_hv = fem.locate_dofs_topological(V.sub(2), fdim, hw)

# hu = 0 on left/right walls  (no normal flux through x = 0, 1)
vw      = mesh.locate_entities_boundary(msh, fdim,
              lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
dofs_hu = fem.locate_dofs_topological(V.sub(1), fdim, vw)

def apply_slip(arr):
    arr[dofs_hv] = 0.0
    arr[dofs_hu] = 0.0

apply_slip(U_h.x.array)

# ── cell sizes h_K = √(area_K) for the CFL condition ────────────────────────
msh.topology.create_connectivity(dim, 0)
c2v   = msh.topology.connectivity(dim, 0)
ncel  = msh.topology.index_map(dim).size_local
geo   = msh.geometry.x
verts = np.array([c2v.links(k) for k in range(ncel)])   # (ncel, 3)
c     = geo[verts]                                        # (ncel, 3, 3)
ab, ac = c[:, 1, :2] - c[:, 0, :2], c[:, 2, :2] - c[:, 0, :2]
h_K   = np.sqrt(0.5 * np.abs(ab[:, 0]*ac[:, 1] - ab[:, 1]*ac[:, 0]))

# ── lumped mass vector (assembled once) ───────────────────────────────────────
one   = fem.Constant(msh, PETSc.ScalarType(np.ones(3)))
M_L   = assemble_vector(form(inner(one, TestFunction(V)) * dx))
M_L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
m_inv = 1.0 / M_L.array

# ── UFL residual (compiled once before the time loop) ────────────────────────
eps   = fem.Constant(msh, PETSc.ScalarType(EPS))
psi   = TestFunction(V)
h_    = U_h[0];  hu_ = U_h[1];  hv_ = U_h[2]
u_    = hu_ / h_;  v_ = hv_ / h_
gh2   = 0.5 * g * h_**2
F1    = [hu_,  hu_*u_ + gh2,  hu_*v_       ]
F2    = [hv_,  hu_*v_,        hv_*v_ + gh2 ]
div_F = sum((F1[i].dx(0) + F2[i].dx(1)) * psi[i] for i in range(3))
visc  = sum(eps * (U_h[i].dx(0)*psi[i].dx(0) +
                   U_h[i].dx(1)*psi[i].dx(1)) for i in range(3))
R_form = form((-div_F - visc) * dx)

# ── helper functions ──────────────────────────────────────────────────────────
def compute_rhs():
    R = assemble_vector(R_form)
    R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return R.array.copy() * m_inv

def dt_cfl():
    """CFL time step: dt = CFL · min_K(h_K / λ_K),  λ_K = |u|+|v|+√(gh)."""
    U   = U_h.x.array.reshape(-1, 3)[verts].mean(axis=1)   # (ncel, 3)
    h_c = np.maximum(U[:, 0], 1e-14)
    u_c = U[:, 1] / h_c;  v_c = U[:, 2] / h_c
    lam = np.abs(u_c) + np.abs(v_c) + np.sqrt(g * h_c)
    return float(CFL * np.min(h_K / (lam + 1e-30)))

def ssp_rk3(dt):
    """Shu–Osher SSP-RK3; slip walls enforced after each stage."""
    u0 = U_h.x.array.copy()

    U_h.x.array[:] = u0 + dt * compute_rhs()
    apply_slip(U_h.x.array);  U_h.x.scatter_forward()

    U_h.x.array[:] = 0.75*u0 + 0.25*(U_h.x.array + dt * compute_rhs())
    apply_slip(U_h.x.array);  U_h.x.scatter_forward()

    U_h.x.array[:] = u0/3 + (2/3)*(U_h.x.array + dt * compute_rhs())
    apply_slip(U_h.x.array);  U_h.x.scatter_forward()

# ── time loop ─────────────────────────────────────────────────────────────────
# U is written as a 3-vector (h, hu, hv).
# In ParaView, Calculator filter gives water height h = U_0, speed = U_1/U_0, etc.
vtk = io.VTKFile(msh.comm, "results/swe_dambreak.pvd", "w")

t             = 0.0
snap_interval = T_END / N_SNAP
next_snap     = snap_interval

vtk.write_function(U_h, t)
print(f"{'step':>6}  {'t':>8}  {'dt':>10}  {'min h':>10}")
step = 0
while t < T_END - 1e-12:
    dt = min(dt_cfl(), T_END - t, next_snap - t + 1e-14)
    ssp_rk3(dt)
    t    += dt
    step += 1
    if t >= next_snap - 1e-12:
        vtk.write_function(U_h, t)
        h_arr = U_h.x.array.reshape(-1, 3)[:, 0]
        print(f"{step:6d}  {t:8.4f}  {dt:10.2e}  {h_arr.min():10.4f}")
        next_snap += snap_interval

vtk.close()
print("Done — open swe_dambreak.pvd in ParaView.")
