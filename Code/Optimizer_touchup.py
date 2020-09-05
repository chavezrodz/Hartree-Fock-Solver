import matplotlib.pyplot as plt
from itertools import product
import itertools
import numpy as np
import os
import glob


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