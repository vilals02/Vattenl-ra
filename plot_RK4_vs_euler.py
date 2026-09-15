import numpy as np 
import matplotlib.pyplot as plt 

CFL = np.array([0.9, 10, 20, 30, 35, 36, 40])
errors_euler = np.array([0.346, 0.346, 0.347, 0.348,
                         0.349, 1.20, 62919933])
errors_rk4 = np.array([0.345, 0.345, 0.345, 0.345,
                       0.345, 0.345, 0.345])

fig, ax = plt.subplots()
ax.plot(CFL, errors_euler, label='Euler')
ax.plot(CFL, errors_rk4, label='RK4')
ax.set_ylim([0.2, 2])
ax.set_xlabel("CFL")
ax.set_ylabel("Error")
ax.set_title("Error as a function of increasing timestep.")
ax.legend()
fig.savefig("plots_for_1-3/error.png")