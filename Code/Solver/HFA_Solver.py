import numpy as np
import scipy.special as sp
from numpy import linalg as LA
import Code.Solver.calculations as calc
from time import time


class HFA_Solver:
    """
    Structural Parameters
    Model_params: Hamiltonian Parameters, must include Filling(float)
    MFP_params: initial guesses for Mean Free Parameters
    """
    def __init__(self, Ham, method='sigmoid', save_seq=False, tol=1e-3, **kwargs):
        self.Hamiltonian = Ham

        # Itteration Method Params
        self.tol = tol
        self.save_seq = save_seq
        self.method = method
        self.__dict__.update(kwargs)
        self.N_digits = int(np.abs(np.log10(self.tol)))

    def Find_filling_lowest_energies(self):
        N_states = np.product(self.Hamiltonian.N_shape)*self.Hamiltonian.mat_dim
        N_occ_states = int(self.Hamiltonian.Filling*N_states)
        Es, k = self.Energies.flatten(), N_occ_states
        indices = np.argpartition(Es, k)[:k]
        indices = np.unravel_index(indices, self.Energies.shape)
        return indices

    def Itteration_Step(self, verbose):
        # Solve Matrix Across all momenta
        self.Energies, self.Eigenvectors = self.Hamiltonian.matrices()
        # Find Indices of all required lowest energies
        indices = self.Find_filling_lowest_energies()
        # Calculate Mean Field Parameters with occupied eigenvectors
        previous_MFP = self.Hamiltonian.MF_params
        occupied_eigenvectors = self.Eigenvectors[indices]
        New_MFP = self.Hamiltonian.Consistency(occupied_eigenvectors)
        # Update Guess
        New_Guess, beta = self.update_guess(New_MFP, previous_MFP)
        self.Hamiltonian.MF_params = New_Guess
        # Logging
        if self.save_seq:
            self.beta_seq.append(beta)
            self.sol_seq.append(New_MFP)
        self.count += 1
        if verbose:
            self.Print_step(New_MFP)
        return New_MFP, New_Guess

    def Itterate(self, verbose=True, save_seq=False):
        t0 = time()

        calc.make_grid(self.Hamiltonian)
        self.Hamiltonian.static_variables()
        self.count = 0
        self.converged = True

        c = self.Hamiltonian.MF_params
        if verbose:
            self.Print_step(c, method='Initial')

        if self.save_seq:
            self.sol_seq = []
            self.beta_seq = []

        a, b = self.Itteration_Step(verbose)

        while LA.norm(a-c) > self.tol:
            c = b
            a, b = self.Itteration_Step(verbose)
            if self.count >= self.Itteration_limit:
                self.converged = False
                break
        self.Hamiltonian.converged = self.converged

        dt = time() - t0
        self.Hamiltonian.Energies, self.Hamiltonian.Eigenvectors = self.Hamiltonian.matrices()
        indices = self.Find_filling_lowest_energies()
        self.Hamiltonian.occupied_energies = self.Energies[indices]
        calc.post_calculations(self.Hamiltonian)

        if self.save_seq:
            self.sol_seq = np.vstack(self.sol_seq)
            self.beta_seq = np.vstack(self.beta_seq)

        if verbose:
            self.Print_step(a, dt, method='Final')

    def Print_step(self, a, t=0, method=None):
        if method is None:
            print('Itteration:', self.count, ' Mean Field parameters:', a.round(self.N_digits))
            return
        elif method == 'Initial':
            print('\nInitial Mean Field parameters:', a.round(self.N_digits))
            return
        elif method == 'Final':
            print(f'Final Mean Field parameter: {a.round(self.N_digits)} Number of itteration steps: {self.count} Time taken:{round(t,3)} \n')
            return

    def update_guess(self, a, b):
        """
        beta is a measure of how fast the mixing goes to zero,
        static for momentum.
        usual values are ~0.5 for exponential, ~3 for sigmoid

        """
        if self.method == 'momentum':
            beta = self.beta
        elif self.method == 'sigmoid':
            beta = sp.expit(-self.count*self.beta/self.Itteration_limit) 
        else:
            print('Error: Itteration Method not found')

        if self.count == 0:
            beta = 0
        elif self.count >= 0.7*self.Itteration_limit and self.count % int(self.Itteration_limit/10) == 0:
            beta = 0.5
        elif self.count == int(0.9*self.Itteration_limit):
            beta = 1

        return (1 - beta)*b + beta*a, beta
