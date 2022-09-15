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
- The exhaustive ansatz search is not automated since it requires node parralelization which is ressource dependent


If you use this code for your own projects, please consider citing the following paper:

```
@article{2021,
	title={Effects of reduced dimensionality, crystal field, electron-lattice coupling, and strain on the ground state of a rare-earth nickelate monolayer},
	volume={104},
	ISSN={2469-9969},
	url={http://dx.doi.org/10.1103/PhysRevB.104.205111},
	DOI={10.1103/physrevb.104.205111},
	number={20},
	journal={Physical Review B},
	publisher={American Physical Society (APS)},
	author={Chavez Zavaleta, Rodrigo and Fomichev, Stepan and Khaliullin, Giniyat and Berciu, Mona},
	year={2021},
	month={Nov}
}
```
