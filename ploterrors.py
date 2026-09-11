import matplotlib.pyplot as plt

errors = [0.879, 0.875, 0.868, 0.869, 0.855, 0.839, 0.784, 0.730, 0.838]
mesh_size = [0.05, 0.06, 0.07, 0.08, 0.12, 0.2, 0.3, 0.4, 0.5]

#p            #mz

#1594        #0.05  
#1149        #0.06
#852         #0.07
#649         #0.08
#331         #0.12
#123         #0.2
#74          #0.3
#41          #0.4
#41          #0.5

plt.loglog(mesh_size, errors)
plt.xlabel(f"Characteristic mesh size [log(h)]")
plt.ylabel(r"$e_h$")
plt.ylim(10**-2, 1)
plt.show()