import numpy as np


def forward_euler_step(M_inv, A, U_n, h):

    f = -1 * M_inv @ A @ U_n
    U_next = U_n + h*f

    return U_next

def rk4_step(M_inv, A, U_n, h):

    f = -1 * M_inv @ A

    k1 = f @ U_n
    k2 = f @ (U_n + k1 * h * 0.5)
    k3 = f @ (U_n + k2 * h * 0.5)
    k4 = f @ (U_n + k3 * h)

    U_next = U_n +  (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return U_next 

def rk3_SSP_step(M_inv, A, U_n, h):

    u1 = forward_euler_step(M_inv, A, U_n, h)
    u2 = 0.75*U_n + 0.25*forward_euler_step(M_inv, A, u1, h)

    U_next = 1/3*U_n + 2/3*forward_euler_step(M_inv, A, u2, h)

    return U_next

def initial_profile_A(x, x10, x20, r0, N):
    init_cond = np.zeros(N)
    for i in range(N):
        x1 = x[0, i]
        x2 = x[1, i]
        C1 = (x1 - x10)**2 + (x2 - x20)**2

        init_cond[i] = (1 - np.tanh(C1/(r0**2) - 1))
        
    return 0.5*init_cond

def initial_profile_B(x, x10, x20, r0, N):
    r0squared = r0 ** 2
    init_cond = np.zeros(N)
    for i in range(N):
        x1 = x[0, i]
        x2 = x[1, i]
        C1 = (x1 - x10)**2 + (x2 - x20)**2
        if C1 <= r0squared:
            init_cond[i] = 1
        else:
            init_cond[i] = 0
        
    return init_cond

def exact_solution_A(x1, x2, x10, x20, r0):

    C1 = (x1 - x10)**2 + (x2 - x20)**2

    init_cond = (1 - np.tanh(C1/(r0**2) - 1))
        
    return 0.5*init_cond

def exact_solution_B(x1, x2, x10, x20, r0):
    r0squared = r0 ** 2
    C1 = (x1 - x10)**2 + (x2 - x20)**2

    if C1 <= r0squared:
        init_cond = 1
    else:
        init_cond = 0
        
    return 0.5*init_cond

def get_speed(x, y):

    return -2*np.pi*y, 2*np.pi*x


def fp(Uh):

    return np.cos(Uh), -1*np.sin(Uh)

def initial_profile_NL(x, N):
    init_cond = np.zeros(N)
    for i in range(N):
        x1 = x[0, i]
        x2 = x[1, i]
        r = x1**2 + x2**2
        if r <= 1:
            init_cond[i] = 14*np.pi/4
        else:
            init_cond[i] = np.pi/4
        
    return init_cond

def conservative_to_primitive(U): 
    gamma = 1.4
    p = (gamma - 1) * (U[:,3] - 0.5 * (U[:,1]**2 + U[:,2]**2) / U[:,0])    
    Up = np.copy(U)
    Up[:,0] = U[:,0]
    Up[:,1] = U[:,1] / U[:,0]
    Up[:,2] = U[:,2] / U[:,0]
    Up[:,3] = p
    return Up

def compute_sound_speed(p, rho):
    gamma = 1.4
    return np.sqrt(gamma * p / rho)

def compute_eps_K(p_n, rho_n, u_n, v_n, C, x, cells):
    """p_n, rho_n, u_n, v_n: (n_nodes,); x: (n_nodes,3); cells: (ncells,3)."""
    eps_K = np.zeros(cells.shape[0])
    for K in range(cells.shape[0]):
        i1, i2, i3 = cells[K]
        p1, p2, p3 = x[i1, :2], x[i2, :2], x[i3, :2]
        h_K = max(np.linalg.norm(p1 - p2),
                  np.linalg.norm(p2 - p3),
                  np.linalg.norm(p1 - p3))
        lam = 0.0
        for i in (i1, i2, i3):
            c = compute_sound_speed(max(p_n[i], 1e-10), max(rho_n[i], 1e-10)) # No div by zero 
            lam = max(lam, np.sqrt(u_n[i]**2 + v_n[i]**2) + c)
        eps_K[K] = C * h_K * lam
    return eps_K

