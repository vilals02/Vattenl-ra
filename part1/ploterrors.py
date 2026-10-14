import matplotlib.pyplot as plt
import numpy as np

def plotfig(data, data_stabilized, init_cond):
  mesh_size = [0.05, 0.08, 0.10, 0.12]
  plt.loglog(mesh_size, data, label="GFEM")
  if data_stabilized is not None:
    plt.loglog(mesh_size, data_stabilized, label="Stabilized GFEM")
  plt.xlabel(f"Characteristic mesh size [log(h)]")
  plt.ylabel(r"$e_h$")
  plt.title(f"Initial condition {init_cond}")
  plt.ylim(10**-4, 1)
  plt.legend()
  plt.savefig(f"errorplots/error_{init_cond}_vs_as.png")
  plt.close()

<<<<<<< Updated upstream
errorsA = [0.341, 0.343, 0.345, 0.348]
errorsB = []
errorsAS = [0.130, 0.133, 0.139, 0.158]
errorsBS = []
=======
errorsA = [0.348, 0.345, 0.342, 0.340]
errorsB = [0.41127981559610904, 0.41038598919932645, 0.42108706484562436, 0.4289585744730342]
errorsAS = [0.000215, 0.000972, 0.00175, 0.00242]
errorsBS = [0.12192088479813963, 0.12404406771107389, 0.13378796782120994, 0.1537117360597822]
>>>>>>> Stashed changes


#p            #mz
#1594        #0.05  
#649         #0.08
#423         #0.10
#331         #0.12

<<<<<<< Updated upstream
plotfig(errorsA, errorsAS, "A")
#plotfig(errorsB, errorsBS, "B")
=======
mesh_size = [0.05, 0.08, 0.10, 0.12]
plt.loglog(mesh_size, errorsB, label="GFEM")
plt.xlabel(f"Characteristic mesh size [log(h)]")
plt.ylabel(r"$e_h$")
plt.title(f"Initial condition B")
plt.ylim(10**-4, 1)
plt.legend()
plt.savefig(f"errorplots/error_B_solo.png")
plt.close()

plotfig(errorsB, errorsBS, "B")
>>>>>>> Stashed changes
