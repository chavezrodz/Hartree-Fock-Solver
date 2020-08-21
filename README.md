Hartree Fock Equations solver, motivated by MIT in Rare-Earth Nickelates
How to use:

1) Declare Hamiltonian, be careful to respect all conditions in the template hamiltonian
2) Chose your model parameters, along with the ones you want to sweep, and the ranges for the MFP guesses write this in the main.py file
3) perform the phase diagram sweep across all guesses, using the slurm script to parralelize
4) run the interpreter to find the optimal guesses across all phase diagram and rerun the phase diagram sweep with the optimal parameters. Feel free to compare the lowest energies with the ones from the interpreter. (main2.py)

The Mean Field Parameters are now printed out across the whole phase region. Plug into an interpreter for your phase diagram
