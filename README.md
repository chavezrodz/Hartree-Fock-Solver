<<<<<<< HEAD
# Hartree-Fock-Solver
<<<<<<< HEAD
=======
>>>>>>> 4dc42d17461e942362c3b5ea5920c966c09646a5
Hartree Fock Equations solver, motivated by MIT in Rare-Earth Nickelates

1) Declare Hamiltonian, be careful to respect all conditions in the template hamiltonian
2) Chose your model parameters, along with the ones you want to sweep, and the ranges for the MFP guesses write this in the PhaseDiagramSweep.py file
3) perform the phase diagram sweep across all guesses, using the slurm script to parralelize
4) run the interpreter to find the optimal guesses across all phase diagram.
5) rerun the phase diagram sweep with the optimal parameters. Feel free to compare the lowest energies with the ones from the interpreter

You now have a Hartree-Fock solution to your phase diagram
<<<<<<< HEAD
>>>>>>> Parrallelizing
=======
>>>>>>> 4dc42d17461e942362c3b5ea5920c966c09646a5
