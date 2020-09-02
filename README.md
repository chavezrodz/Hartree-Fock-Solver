Hartree Fock Equations solver, motivated by MIT in Rare-Earth Nickelates
How to use:

1) Declare Hamiltonian, be careful to respect all conditions in the template hamiltonian
2) Chose your model parameters, along with the ones you want to sweep, and the ranges for the MFP guesses write this in the params.py file
3) submit the trials slurm script to parralelize guesses
4) submit the final slurm script to find the optimal guesses across all phase diagram and rerun the phase diagram sweep with the optimal parameters.

The Mean Field Parameters are now printed out across the whole phase region. Plug into an interpreter for your phase diagram
