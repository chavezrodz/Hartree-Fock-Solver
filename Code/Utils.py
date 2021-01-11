import numpy as np
import os
import itertools


def tuplelist(x):
    LIST = []
    for v in itertools.product(*x):
        LIST.append(v)
    return LIST


def Read_MFPs(folder):
    N = len(os.listdir(folder))
    for i in range(N):
        file = os.path.join(folder,'MF'+str(i)+'.csv')
        if i == 0:
            MF = np.loadtxt(file, delimiter=',')
        else:
            MF = np.dstack((MF, np.loadtxt(file,delimiter=',')))
    return MF