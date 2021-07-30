# Hartree Fock Itterator motivated by MIT in rare earth Nickelates

This is the code used to produce the article ["Effects of reduced dimensionality, crystal field, electron-lattice coupling, and strain on the ground-state of a rare-earth nickelates monolayer"](https://arxiv.org/abs/2106.12644)'s results.

## Features:
- HFA Solver: iterates MFP guesses based on dynamic ansatz techniques
- Diagram sweeper to solve for phase diagrams
- 1D cut to study the effect of a single parameter
- Meta Diagram: To track the change of one phase with two parameters varying

## Single point itteration:
(most with or without HFA, arbitrary BZ code in progress)
- Fermi surface(2D)
- Bandstructure spaghetti plot
- convergence tracking of the mfp evolution
- state projections on density of states

## To use:
Execute any of the runner_*.py files changing parameters within. Any new hamiltonian class could be added to the models folder

### Caveats:
- Most alternative optimizers techniques are not finished given that they are not needed to achieve ~99% convergence
- The 1D cut is out of date since it was not used in the final publication
- The exhaustive ansatz search is not automated since it requires node parralelization which is ressource dependent
