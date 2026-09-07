import numpy as np


def forward_euler_step(M_inv, C, U_n, h):

    f = M_inv @ C @ U_n
    U_next = U_n - h*f

    return U_next

def initial_profile(x, x10, x20, r0, N):
    init_cond = np.zeros(N)
    for i in range(N):
        x1 = x[0, i]
        x2 = x[1, i]
        C1 = (x1 - x10)**2 + (x2 - x20)**2

        init_cond[i] = (1 - np.tanh(C1/(r0**2) - 1))
        
    return 0.5*init_cond

def get_speed(x, y):

    return -2*np.pi*y, 2*np.pi*x
