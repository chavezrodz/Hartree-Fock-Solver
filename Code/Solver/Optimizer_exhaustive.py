from itertools import product
import itertools
import numpy as np
import os
import glob

def Read_MFPs(folder):
    N = len(os.listdir(folder))
    for i in range(N):
        file = os.path.join(folder,'MF'+str(i)+'.csv')
        if i ==0:
            MF = np.loadtxt(file,delimiter=',')
        else:
            MF = np.dstack((MF,np.loadtxt(file,delimiter=',')))
    return MF

def Optimizer_exhaustive(Input_Folder, params_list, input_MFP=False, verbose=False):
    """
    Input list of arrays of energy across phase region,
    return best guess per region
    """
    folderlist = []
    for i in range(len(params_list)):
        folderlist.append(os.path.join(Input_Folder,'Guess'+str(np.array(params_list[i]))))

    # Stack all energies,convergence arrays
    E_Tower,C_Tower = [],[]
    for i, folder in enumerate(folderlist):
        E_file = os.path.join(folder,'Energies.csv')
        C_file = os.path.join(folder,'Convergence_Grid.csv')
        
        E_Tower.append(np.loadtxt(E_file,delimiter=','))
        C_Tower.append(np.loadtxt(C_file,delimiter=','))
    E_Tower,C_Tower = np.dstack(E_Tower), np.dstack(C_Tower).astype(bool)

    if input_MFP:
        Solutions = []
        for i, folder in enumerate(folderlist):
            Solutions.append(Read_MFPs(os.path.join(folder,'MF_Solutions')))
        Solutions = np.transpose(np.stack(Solutions,axis=-1),(0,1,3,2))
        Unconverged_Sols = np.empty(Solutions.shape)
        Unconverged_Sols[:] = np.nan

    # Find Indices of lowest energies across stack
    ind = np.argmin(E_Tower,axis=-1)

    # Lowest achievable energy
    Optimal_Energy = np.min(E_Tower,axis=-1)
    Optimal_Convergence = np.min(C_Tower,axis=-1)

    # Recover best guess across phase diagram
    Diag_Shape = E_Tower.shape[:-1]
    Optimal_Guesses = np.zeros((*Diag_Shape,len(params_list[0])))

    print(Solutions.shape)
    i,j = np.indices(Diag_Shape,sparse=True)
    i,j = i.flatten(),j.flatten()
    for v in itertools.product(i,j):
        Optimal_Guesses[v] = np.array(params_list[ind[v]])
        if verbose:
            print('i ind:',v[0],'j ind:',v[1],'Best guess:', params_list[ind[v]])
        if input_MFP:
            for k in range(len(params_list)):
                if not C_Tower[v][k]:
                    Unconverged_Sols[v][k] = Solutions[v][k]
                    Solutions[v][k] = np.nan

    print(Unconverged_Sols[2,3])
    print(Solutions[2,3])
    print(E_Tower[2,3])

    return Optimal_Guesses, Optimal_Energy
