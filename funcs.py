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


