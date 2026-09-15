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

errorsA = [0.341, 0.343, 0.345, 0.348]
errorsB = []
errorsAS = [0.130, 0.133, 0.139, 0.158]
errorsBS = []


#p            #mz
#1594        #0.05  
#649         #0.08
#423         #0.10
#331         #0.12

plotfig(errorsA, errorsAS, "A")
#plotfig(errorsB, errorsBS, "B")