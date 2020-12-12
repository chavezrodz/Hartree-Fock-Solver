from time import time
import numpy as np
import itertools
import sys
import os
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Display.Itteration_sequence import Itteration_sequence
import Code.Display.DispersionRelation as DR
import matplotlib.pyplot as plt

Model_Params = dict(
    N_shape=(15, 15),
)

# bandwidth is 3 or everything zero
# Code
a = time()
# MF_params = np.zeros(5)
# MF_params = np.array([ 0. , 0.9 ,0. , 0. , 0.])
MF_params = np.array([ 0.9 , 0.9 , 0. , 0.9 , 0.])
# MF_params = np.array([ 0.924, -0.934,  0.817, -0.668, -0.02 ])
# MF_params = np.array([ 0.758, -0.712, -0.111, -0.642,  0.056])
# MF_params = np.array([ 0.255, -0.712, -0.384, -0.914,  0.904])
# MF_params = np.array([-0.201,  0.904,  0.169, -0.318, -0.723])
# MF_params = np.array([ 0.708,  0.737, -0.094,  0.647,  0.102])
# MF_params = np.array([ 0.922,  0.995, -0.005,  0.92,   0.046])# + np.random.rand(5) THIS ONE NEVER CONVERGES
# MF_params = np.array([0,-0.18,-0.36,0,0])
# MF_params = np.array([-0.911,  1.,    -0.,    -0.911, -0.055])

# MF_params = np.random.rand(5)*2 -1

Model = Hamiltonian(Model_Params, MF_params)

Solver = HFA_Solver(Model, method='sigmoid', beta=1, Itteration_limit=50, tol=1e-3, save_seq=True)

Solver.Itterate(verbose=True)

# print('occupied states', len(Solver.indices))
# print('total states', Solver.Energies.shape)
# print('energies up to Fermi', sorted(Solver.occupied_energies)[1240:])
# print('all energies', sorted(Solver.Energies.flatten())[1240:1260])
# Itteration_sequence(Solver)
# DR.DispersionRelation_Zcut(Solver)
# DR.DOS(Solver)

# 1-z-up, 2-z-up, 1-zbar-up, 2-zbar-up

a = np.array([1.,0,0,0,0,0,0,0])
b = np.array([0,1.,0,0,0,0,0,0])
c = np.array([0,0,1.,0,0,0,0,0])
d = np.array([0,0,0,1.,0,0,0,0])
e = np.array([0,0,0,0,1.,0,0,0])
f = np.array([0,0,0,0,0,1.,0,0])
g = np.array([0,0,0,0,0,0,1.,0])
h = np.array([0,0,0,0,0,0,0,1.])

z2_projectors = [a,b,e,f]
x2my2_projectors = [c,d,g,h]
spin_up = [a,b,c,d]
spin_down = [e,f,g,h]
site1 = [a,c,e,g]
site2 = [b,d,f,h]

z2_energies = []
x2my2_energies = []
spin_up_energies = []
spin_down_energies = []
site1_energies = []
site2_energies = []

print(Solver.Conductor)
print('energy is ', Solver.Final_Total_Energy)
print('eigenvectors are')
print(Solver.Eigenvectors.shape)
print(Solver.Energies.shape)
for i, item in enumerate(Solver.Eigenvectors):
    for j, elem in enumerate(item):
        for jp, elem3 in enumerate(elem):
            elem_tr = elem3.transpose()
            for k, elem2 in enumerate(elem_tr):
                # print('eigenstate', elem2.round(decimals=2))
                # print('eigenstate norm', np.dot(np.conj(elem2), elem2))
                energy_elem = Solver.Energies[i,j,jp,k]
                # print('energy', energy_elem)

                z2_elem = (np.sum([np.dot(np.conj(elem2),proj)**2 \
                                    for proj in z2_projectors]).round(decimals=1)).real
                z2_energies.append([energy_elem,z2_elem])
                # don't multiply the energy by it, add it as a weight for the histogram

                x2my2_elem = (np.sum([np.dot(np.conj(elem2),proj)**2 \
                                for proj in x2my2_projectors]).round(decimals=1)).real
                x2my2_energies.append([energy_elem,x2my2_elem])

                spin_up_elem = (np.sum([np.dot(np.conj(elem2),proj)**2 \
                                for proj in spin_up]).round(decimals=1)).real
                spin_up_energies.append([energy_elem,spin_up_elem])

                spin_down_elem = (np.sum([np.dot(np.conj(elem2),proj)**2 \
                                for proj in spin_down]).round(decimals=1)).real
                spin_down_energies.append([energy_elem,spin_down_elem])

                site1_elem = (np.sum([np.dot(np.conj(elem2),proj)**2 \
                                for proj in site1]).round(decimals=1)).real
                site1_energies.append([energy_elem,site1_elem])

                site2_elem = (np.sum([np.dot(np.conj(elem2),proj)**2 \
                                for proj in site2]).round(decimals=1)).real
                site2_energies.append([energy_elem,site2_elem])

                # print('3z^2 - r^2 orbital character', z2_elem)
                # print('x^2 - y^2 orbital character', x2my2_elem)
                # print('spin up character', spin_up_elem)
                # print('spin down character', spin_down_elem)
# print(z2_energies)

z2_energies = np.array(sorted(z2_energies,key=lambda x: x[0]))
x2my2_energies = np.array(sorted(x2my2_energies,key=lambda x: x[0]))
spin_up_energies = np.array(sorted(spin_up_energies,key=lambda x: x[0]))
spin_down_energies = np.array(sorted(spin_down_energies,key=lambda x: x[0]))
site1_energies = np.array(sorted(site1_energies,key=lambda x: x[0]))
site2_energies = np.array(sorted(site2_energies,key=lambda x: x[0]))

# print(z2_energies)


# fig0, ax = plt.subplots(1,1)
all_DOS = np.histogram(Solver.Energies.flatten(), bins = 'fd')
# ax.set_ylim(0, top_cutoff)
bins = all_DOS[1]
top_cutoff = 1000
top_text_pos = 800
fig, [[ax1, ax2], [ax3, ax4], [ax5,ax6]] = plt.subplots(3,2)
ax1.set_title(r'$3z^2-r^2$ orbital DOS')
ax1.axvline(Solver.Fermi_Energy, label='Fermi Energy',color='red')
# let's also plot the counts below the fermi level
ax1.text(0.,top_text_pos, 'occupied states = {}'.format((np.sum([weight for (elem,weight) in \
                    z2_energies if elem < Solver.Fermi_Energy])).real.round(decimals=0)), ha = 'center' )
z2_DOS = ax1.hist(z2_energies[:,0], bins=bins, weights = \
                                                    z2_energies[:,1])
ax1.set_ylim(0, top_cutoff)
ax2.set_title(r'$x^2-y^2$ orbital DOS')
ax2.axvline(Solver.Fermi_Energy, label='Fermi Energy',color='red')
ax2.text(0.,top_text_pos, 'occupied states = {}'.format((np.sum([weight for (elem,weight) in \
                    x2my2_energies if elem < Solver.Fermi_Energy])).real.round(decimals=0)), ha = 'center' )
x2my2_DOS = ax2.hist(x2my2_energies[:,0], bins=bins, weights = \
                                                x2my2_energies[:,1])
ax2.set_ylim(0, top_cutoff)
ax3.set_title(r'Spin $\uparrow$ DOS')
ax3.axvline(Solver.Fermi_Energy, label='Fermi Energy',color='red')
ax3.text(0.,top_text_pos, 'occupied states = {}'.format((np.sum([weight for (elem,weight) in \
                    spin_up_energies if elem < Solver.Fermi_Energy])).real.round(decimals=0)), ha = 'center' )
spin_up_DOS = ax3.hist(spin_up_energies[:,0], bins=bins, weights = \
                                            spin_up_energies[:,1])
ax3.set_ylim(0, top_cutoff)
ax4.set_title(r'Spin $\downarrow$ DOS')
ax4.axvline(Solver.Fermi_Energy, label='Fermi Energy',color='red')
ax4.text(0.,top_text_pos, 'occupied states = {}'.format((np.sum([weight for (elem,weight) in \
                    spin_down_energies if elem < Solver.Fermi_Energy])).real.round(decimals=0)), ha = 'center' )
spin_down_DOS = ax4.hist(spin_down_energies[:,0], bins=bins, weights = \
                                            spin_down_energies[:,1])
ax4.set_ylim(0, top_cutoff)

ax5.set_title(r'Homogeneous DOS')
ax5.axvline(Solver.Fermi_Energy, label='Fermi Energy',color='red')
ax5.text(0.,top_text_pos, 'occupied states = {}'.format((np.sum([weight for (elem,weight) in \
                    site1_energies if elem < Solver.Fermi_Energy])).real), ha = 'center' )
site1_DOS = ax5.hist(site1_energies[:,0], bins=bins, weights = \
                                            site1_energies[:,1])
ax5.set_ylim(0, top_cutoff)
ax6.set_title(r'Site-alternating DOS')
ax6.axvline(Solver.Fermi_Energy, label='Fermi Energy',color='red')
ax6.text(0.,top_text_pos, 'occupied states = {}'.format((np.sum([weight for (elem,weight) in \
                    site2_energies if elem < Solver.Fermi_Energy])).real.round(decimals=0)), ha = 'center' )
site2_DOS = ax6.hist(site2_energies[:,0], bins=bins, weights = \
                                            site2_energies[:,1])
ax6.set_ylim(0, top_cutoff)
plt.tight_layout()
plt.show()

print(Solver.bandwidth_calculation())
