import Code.Nickelates.Interpreter as In
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

def fill_nans_nearest(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:,None], idx]

def Optimizer_touchup(MFPs, Convergence_Grid):
    """
    Input list of arrays of energy across phase region,
    return best guess per region
    """
    # Indices where convergence failed
    indices = np.where(Convergence_Grid==0,)
    indices = np.transpose(np.stack(indices))
    indices = list(map(tuple,indices))
    
    # Replace unconverged with Nans
    for v in indices:
        MFPs[v] = np.nan

    # Replace Nans with nearest neighbours
    for i in range(5):
        MFPs[:,:,i] = fill_nans_nearest(MFPs[:, :, i])
    return MFPs

def Optimizer_smoothing(mfps,sigma=[1,1]):
    y = np.zeros(mfps.shape)
    for i in range(mfps.shape[2]):
        y[:,:,i] = sp.ndimage.filters.gaussian_filter(mfps[:,:,i], sigma, mode='nearest')
    return y

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
        C_file = os.path.join(folder,'Convergence.csv')
        
        E_Tower.append(np.loadtxt(E_file,delimiter=','))
        C_Tower.append(np.loadtxt(C_file,delimiter=','))
    E_Tower,C_Tower = np.dstack(E_Tower), np.dstack(C_Tower).astype(bool)

    if input_MFP:
        Solutions = []
        for i, folder in enumerate(folderlist):
            MFPs = Read_MFPs(os.path.join(folder,'MF_Solutions'))
            Phase = In.array_interpreter(MFPs)
            MF_Spin_orb = Phase[:,:,1:]
            state = In.arr_to_int(MF_Spin_orb)
            Solutions.append(state)
        Solutions = np.stack(Solutions,axis=-1)
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
                    Solutions[v][k] = -1
    return Optimal_Guesses, Optimal_Energy
