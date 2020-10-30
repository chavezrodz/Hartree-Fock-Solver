import numpy as np
import itertools
import sys
import os
from time import time
import argparse
import logging
from Code.Solver.HFA_Solver import HFA_Solver
from Code.Utils.tuplelist import tuplelist as tp
from Code.Solver.PhaseDiagramSweeper import Phase_Diagram_Sweeper
from Code.Nickelates.Hamiltonian import Hamiltonian
from Code.Solver.Optimizer_exhaustive import Optimizer_exhaustive
from Code.Display.DiagramPlots import DiagramPlots
from Code.Display.PhasePlots import PhasePlots

i,j = 'U','J',
i_values = np.linspace(0,6,10)
j_values = np.linspace(0,3,10)

folders = os.listdir(os.path.join('Results','Results'))
for folder in folders:
	frf = os.path.join('Results','Results',folder,'Final_Results')
	print(frf)
	PhasePlots(i,i_values,j,j_values,frf)
