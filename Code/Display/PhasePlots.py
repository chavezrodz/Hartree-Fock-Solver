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
# plt.plot()
# plt.close()
# Magnetization


MF_Spin = Phase[:,:,1:3]
Spin_Dict = {0:'\downarrow', 1:'0',2:r'\uparrow'}

states = MF_Spin.reshape(-1,MF_Spin.shape[-1])
states = np.unique(MF_Spin.reshape(-1,MF_Spin.shape[-1]),axis=0)
states = list(states)

labels = [r'$'+Spin_Dict[state[0]]+Spin_Dict[state[1]]+'$' for state in states]

# IDS
# MF_Spin[:,:,0] = 
print(labels)

mag = ax.pcolormesh(MF_Spin[:,:,0].T,alpha=1)
# (?# state_names = [r'$0 0$', r'$0 \uparrow$', r'$0 \downarrow$', r'$\uparrow 0$', r'$\uparrow \uparrow$', r'$\uparrow \downarrow$', r'$\downanp.rrow 0$', r'$\downarrow \uparrow$', r'$\downarrow \downarrow$'])
# print(states)

ep.draw_legend(mag, classes = states,titles=labels)#,
    # titles=[r'$0 0$', r'$0 \uparrow$', r'$0 \downarrow$', r'$\uparrow 0$', r'$\uparrow \uparrow$', r'$\uparrow \downarrow$', r'$\downarrow 0$', r'$\downarrow \uparrow$', r'$\downarrow \downarrow$'],)
    # classes=[0, 1, 2, 3,4,5,6,7,8]
    # )
# Orbit_Dict ={0:r'$ 0 $',1:r'$ z$',2:r'$\downarrow$'}

# # Orbital order
# OD = Phase[:,:,2]
# im = ax.pcolormesh(OD,alpha=1)

# ep.draw_legend(im,
#     titles=[r'$0 0$', r'$0 z$', r'$0 \bar{z}$', r'$z 0$', r'$z z$', r'$z \bar{z}$', r'$\bar{z} 0$', r'$\bar{z} z$', r'$\bar{z} \bar{z}$'],
#     classes=[0, 1, 2, 3,4,5,6,7,8])

plt.tight_layout()
plt.show()
