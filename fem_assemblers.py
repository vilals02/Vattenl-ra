
import numpy as np
from funcs import get_speed
from scipy.sparse import coo_matrix


def hat_gradients(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    detJ = ((x[1] - x[0]) * (y[2] - y[0])
            - (x[2] - x[0]) * (y[1] - y[0]))

    if abs(detJ) < 1.0e-15:
        raise ValueError("Degenerate triangle.")

    area = 0.5 * abs(detJ)

    b = np.array([y[1] - y[2],
                  y[2] - y[0],
                  y[0] - y[1]], dtype=float) / detJ

    c = np.array([x[2] - x[1],
                  x[0] - x[2],
                  x[1] - x[0]], dtype=float) / detJ

    return area, b, c


def create_eps_K_array(p, t):

    eps_K = np.zeros(t.shape[1])

    for K in range(t.shape[1]):
        h_K = -1 

        i1 = t[0,K]
        i2 = t[1,K]
        i3 = t[2,K]

        # points of triangle K
        p1 = np.array([p[0, i1], p[1, i1]])
        p2 = np.array([p[0, i2], p[1, i2]])
        p3 = np.array([p[0, i3], p[1, i3]])

        # distances
        d13 = np.linalg.norm(p1 - p3, 2)
        d23 = np.linalg.norm(p2 - p3, 2)
        d12 = np.linalg.norm(p1 - p2, 2)

        # velocities
        vx1, vy1 = get_speed(p1[0], p1[1])
        vx2, vy2 = get_speed(p2[0], p2[1])
        vx3, vy3 = get_speed(p3[0], p3[1])

        # max distance of the triangle
        max_dist = max(d13, d23, d12)
        if (max_dist) > h_K:
            h_K = max_dist

        beta_1 = np.linalg.norm(np.array([vx1, vy1]), 2)
        beta_2 = np.linalg.norm(np.array([vx2, vy2]), 2)
        beta_3 = np.linalg.norm(np.array([vx3, vy3]), 2)

        eps_K[K] = h_K * max(np.array([beta_1, beta_2, beta_3]))

    return 0.5 * eps_K


def _assemble_sparse(npnt, t, local_matrix):
    nt = t.shape[1] # Iterate through triangles 

    rows = []
    cols = []
    vals = []

    for K in range(nt):
        loc2glb = t[:, K]
        AK = local_matrix(K, loc2glb)

        for i in range(3):
            for j in range(3):
                rows.append(loc2glb[i])
                cols.append(loc2glb[j])
                vals.append(AK[i, j])

    return coo_matrix((vals, (rows, cols)),
                      shape=(npnt, npnt)).tocsr()

def _assemble_stiffness_sparse(npnt, p, t, local_matrix):
    nt = t.shape[1] # Iterate through triangles 

    rows = []
    cols = []
    vals = []

    eps = create_eps_K_array(p, t)

    for K in range(nt):
        loc2glb = t[:, K]
        AK = eps[K] * local_matrix(K, loc2glb)

        for i in range(3):
            for j in range(3):
                rows.append(loc2glb[i])
                cols.append(loc2glb[j])
                vals.append(AK[i, j])

    return coo_matrix((vals, (rows, cols)),
                      shape=(npnt, npnt)).tocsr()

def mass_assembler_2d(p, t):
    """
    Assemble M_ij = int phi_j phi_i dx.

    Local matrix:
        M_K = |K|/12 [[2,1,1],
                      [1,2,1],
                      [1,1,2]]
    """
    npnt = p.shape[1]

    M0 = np.array([[2., 1., 1.],
                   [1., 2., 1.],
                   [1., 1., 2.]]) / 12.0

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        area, _, _ = hat_gradients(x, y)
        return area * M0

    return _assemble_sparse(npnt, t, local_matrix)


def load_assembler_2d(p, t, f):
    """
    Assemble b_i = int f phi_i dx using the corner quadrature rule
    from Larson-Bengzon.
    """
    npnt = p.shape[1]
    nt = t.shape[1]

    b_global = np.zeros(npnt)

    for K in range(nt):
        loc2glb = t[:, K]

        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, _, _ = hat_gradients(x, y)

        bK = area / 3.0 * np.array([
            f(x[0], y[0]),
            f(x[1], y[1]),
            f(x[2], y[2])
        ], dtype=float)

        b_global[loc2glb] += bK

    return b_global


def stiffness_assembler_2d(p, t):
    """
    Assemble A_ij = int a grad(phi_j).grad(phi_i) dx.

    The coefficient a is evaluated at the triangle centroid,
    as in Larson-Bengzon.
    """
    npnt = p.shape[1]

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, b, c = hat_gradients(x, y)

        xc = np.mean(x)
        yc = np.mean(y)
        # abar = float(a(xc, yc)) # Larson-Bengzon

        return area * (np.outer(b, b) + np.outer(c, c))

    return _assemble_stiffness_sparse(npnt, p, t, local_matrix)


def convection_assembler_2d(p, t, bx, by):
    """
    Assemble

        C_ij = int (beta . grad(phi_j)) phi_i dx,

    following Larson-Bengzon.

    bx and by are nodal arrays.
    """
    npnt = p.shape[1]

    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)

    if bx.shape != (npnt,) or by.shape != (npnt,):
        raise ValueError("bx and by must have shape (p.shape[1],).")

    def local_matrix(K, loc2glb):
        x = p[0, loc2glb]
        y = p[1, loc2glb]

        area, b, c = hat_gradients(x, y)

        bxmid = np.mean(bx[loc2glb])
        bymid = np.mean(by[loc2glb])

        beta_grad_phi = bxmid * b + bymid * c

        return area / 3.0 * np.outer(np.ones(3), beta_grad_phi)

    return _assemble_sparse(npnt, t, local_matrix)
