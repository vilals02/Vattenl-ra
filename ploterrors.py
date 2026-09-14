import matplotlib.pyplot as plt

errorsA = [0.348, 0.345, 0.342, 0.340]
errorsB = []
errorsAS = []
errorsBS = []
mesh_size = [0.05, 0.08, 0.10, 0.12]


#p            #mz
#1594        #0.05  
#649         #0.08
#            #0.10
#331         #0.12

plt.loglog(mesh_size, errors)
plt.xlabel(f"Characteristic mesh size [log(h)]")
plt.ylabel(r"$e_h$")
plt.ylim(10**-2, 1)
plt.show()