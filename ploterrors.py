import matplotlib.pyplot as plt

errorsA = [0.348, 0.345, 0.342, 0.340]
errorsB = [0.433, 0.422, 0.417, 0.422]
errorsAS = [0.000215, 0.000972, 0.00175, 0.00242]
errorsBS = [0.000167, 0.000764, 0.00157, 0.00205]
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