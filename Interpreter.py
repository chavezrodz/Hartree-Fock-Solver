import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import scipy.interpolate

# Input arrays of two parameters and phase

MF = np.zeros((4,10,10))

Results_Folder = 'Results'

for i in range(4):
	MF[i,:,:] = np.loadtxt(Results_Folder+'/Final_Results/MF_Solutions/MF'+str(i)+'.csv',delimiter=",")

# MF = np.random.rand(3,20,20)

# print(MF)


Dict ={0:'Charge Modulation',1:'Ferromagnetism', 2:'Orbital Disproportionation',3:'Anti Ferromagnetism'}

# MF = np.rint(MF)
for i in range(4):
	plt.pcolormesh(np.abs(MF[i]).T)
	plt.title(Dict[i])
	plt.xlabel('$U/t_{1}$')
	plt.ylabel('$J/t_{1}$')
	plt.xticks(np.linspace(0, 10, 4),np.arange(0,8,2))
	plt.yticks(np.linspace(0, 10, 4),np.arange(0,4,1))	
	plt.colorbar()
	plt.savefig(Results_Folder+'/Final_Results/Plots/'+Dict[i]+'.png',transparent=True)
	plt.show()
"""
"""
plt.title('Convergence Grid')
plt.pcolormesh(np.loadtxt(Results_Folder+'/Final_Results/Convergence_Grid.csv',delimiter=','),cmap='gray')
plt.xlabel('$U/t_{1}$')
plt.ylabel('$J/t_{1}$')
plt.xticks(np.linspace(0, 10, 4),np.arange(0,8,2))
plt.yticks(np.linspace(0, 10, 4),np.arange(0,4,1))
plt.savefig(Results_Folder+'/Final_Results/Plots/Convergence_Grid.png',transparent=True)
plt.show()

