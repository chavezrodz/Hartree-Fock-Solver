import os
import itertools
import earthpy.plot as ep
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate
import Code.Nickelates.Interpreter as In

from Code.Utils.Read_MFPs import Read_MFPs

MF = Read_MFPs(os.path.join('Results','Run_sep_20','Final_Results','MF_Solutions'))

Phase = In.array_interpreter(MF)

f, ax = plt.subplots(figsize=(8,5))
ax.set_xlabel('$U/t_{1}$')
ax.set_ylabel('$J/t_{1}$')
plt.xticks(np.linspace(0, MF.shape[0], 4),np.arange(0,8,2))
plt.yticks(np.linspace(0, MF.shape[1], 4),np.arange(0,4,1))	

CM = Phase[:,:,0]
# Charge Modulation
CS = ax.contour(CM.T,colors='red',levels=[0.1,0.3,0.5])
ax.clabel(CS, inline=True, fontsize=10)

# Magnetization
MF_Spin = Phase[:,:,1]
mag = ax.pcolormesh(MF_Spin.T,alpha=0.5)
ep.draw_legend(mag,
    titles=[r'$0 0$', r'$\uparrow 0$', r'$\uparrow \downarrow$', r'$\uparrow \uparrow$'],
    classes=[0, 1, 2, 3])

# Orbital order
OD = Phase[:,:,2]
im = ax.pcolormesh(OD,alpha=0.5)
ep.draw_legend(im,
    titles=[r'$\bar{Z}$', r'$\bar{z}$', r'$z$', r'$Z$'],
    classes=[0, 1, 2, 3])
plt.tight_layout()
plt.show()


