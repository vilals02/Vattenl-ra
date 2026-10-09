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
from mesh import dolfinx_to_pet
from dolfinx.fem.petsc import assemble_vector
from dolfinx.fem import form
from ufl import inner, dx, TestFunction
from petsc4py import PETSc
import funcs as funky

# ── parameters ───────────────────────────────────────────────────────────────
g      = 9.81    # gravity
CFL    = 0.4     # CFL number
T_END  = 0.2   # final time
N_SNAP = 20      # snapshots written to disk

# ── mesh ─────────────────────────────────────────────────────────────────────
nx, ny = 200, 40
msh = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [1.0, 0.2]],
    [nx, ny],
    mesh.CellType.triangle,
)
pts, _, tri = dolfinx_to_pet(msh)
dim  = msh.topology.dim
fdim = dim - 1

# ── function space ────────────────────────────────────────────────────────────
V   = fem.functionspace(msh, ("Lagrange", 1, (4,)))   # U = (rho, rho * u, rho * v, E)
U_h = fem.Function(V, name="U")
Q0      = fem.functionspace(msh, ("DG", 0))
eps_f   = fem.Function(Q0, name="eps")
cells_Q = Q0.dofmap.list[:, 0]      # one dof per cell

# ── initial condition: dam break ──────────────────────────────────────────────
x   = V.tabulate_dof_coordinates()       # (n_dofs, 4)
cells_V = V.dofmap.list # (cells, 4)
arr = U_h.x.array.reshape(-1, 4)
L   = x[:, 0] < 0.5                     # left half: deep water
arr[L,  0] = 1.0                           # rho = 1  (left)
arr[~L, 0] = 0.125                         # rho = 0.125 (right)
arr[L, 3] = 2.5
arr[~L, 3] = 0.25
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
one   = fem.Constant(msh, PETSc.ScalarType(np.ones(4)))
M_L   = assemble_vector(form(inner(one, TestFunction(V)) * dx))
M_L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
m_inv = 1.0 / M_L.array

# ── helper functions ──────────────────────────────────────────────────────────
def compute_rhs(R_form):
    R = assemble_vector(R_form)
    R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return R.array.copy() * m_inv

def dt_cfl():
    """CFL time step: dt = CFL · min_K(h_K / λ_K),  λ_K = |u|+|v|+√(gh)."""
    U   = U_h.x.array.reshape(-1, 4)[verts].mean(axis=1)   # (ncel, 3)
    h_c = np.maximum(U[:, 0], 1e-14)
    u_c = U[:, 1] / h_c;  v_c = U[:, 2] / h_c
    lam = np.abs(u_c) + np.abs(v_c) + np.sqrt(g * h_c)
    return float(CFL * np.min(h_K / (lam + 1e-30)))

def ssp_rk3(dt, R_form):
    """Shu–Osher SSP-RK3; slip walls enforced after each stage."""
    u0 = U_h.x.array.copy()

    U_h.x.array[:] = u0 + dt * compute_rhs(R_form)
    apply_slip(U_h.x.array);  U_h.x.scatter_forward()

    U_h.x.array[:] = 0.75*u0 + 0.25*(U_h.x.array + dt * compute_rhs(R_form))
    apply_slip(U_h.x.array);  U_h.x.scatter_forward()

    U_h.x.array[:] = u0/3 + (2/3)*(U_h.x.array + dt * compute_rhs(R_form))
    apply_slip(U_h.x.array);  U_h.x.scatter_forward()

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

# ── UFL residual (compiled once before the time loop) ────────────────────────
C = 1 #parameter
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

# ── time loop ─────────────────────────────────────────────────────────────────
# U is written as a 4-vector (rho, rho * u, rho * v, E).
# In ParaView, Calculator filter gives water height rho = U_0, speed = U_1/U_0, etc.
vtk = io.VTKFile(msh.comm, "results/swe_dambreak.pvd", "w")

t             = 0.0
snap_interval = T_END / N_SNAP
next_snap     = snap_interval

vtk.write_function(U_h, t)
print(f"{'step':>6}  {'t':>8}  {'dt':>10}  {'min h':>10}")
step = 0
while t < T_END - 1e-12:
    dt = min(dt_cfl(), T_END - t, next_snap - t + 1e-14)
    # Up = funky.conservative_to_primitive(U_h)
    update_eps()
    ssp_rk3(dt, R_form)
    t    += dt
    step += 1
    if t >= next_snap - 1e-12:
        vtk.write_function(U_h, t)
        h_arr = U_h.x.array.reshape(-1, 4)[:, 0]
        print(f"{step:6d}  {t:8.4f}  {dt:10.2e}  {h_arr.min():10.4f}")
        next_snap += snap_interval

vtk.close()
print("Done — open swe_dambreak.pvd in ParaView.")
