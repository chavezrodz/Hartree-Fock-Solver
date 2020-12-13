import numpy as np
import scipy.special as sp
from numpy import linalg as LA


class HFA_Solver:
    """
    Structural Parameters
    Model_params: Hamiltonian Parameters, must include Filling(float)
    MFP_params: initial guesses for Mean Free Parameters
    """
    def __init__(self, Ham, method='momentum', beta=0.7, Itteration_limit=50, tol=1e-3, save_seq=False):
        self.Hamiltonian = Ham

        # Itteration Method Params
        self.beta = beta
        self.Itteration_limit = Itteration_limit
        self.tol = tol
        self.method = method
        self.save_seq = save_seq

        self.N_states = Ham.N_states
        self.N_occ_states = int(Ham.Filling*self.N_states)
        self.N_params = len(Ham.MF_params)
        self.N_digits = int(np.abs(np.log10(self.tol)))
        self.sub_params = np.zeros((self.N_params, self.N_occ_states))

    def Find_filling_lowest_energies(self):
        Initial_shape = self.Energies.shape
        k = self.N_occ_states
        Ec = self.Energies.flatten()
        indices = np.argpartition(Ec, k)[:k]
        indices = np.unravel_index(indices, shape=Initial_shape)
        self.indices_array = indices
        indices = np.transpose(np.stack(indices))
        self.indices = list(map(tuple, indices))

    def Print_step(self, a, method=None):
        if method is None:
            print('Itteration:', self.count, ' Mean Field parameters:', a.round(self.N_digits))
            return
        elif method == 'Initial':
            print('\nInitial Mean Field parameters:', a.round(self.N_digits))
            return
        elif method == 'Final':
            print('Final Mean Field parameter:', a.round(self.N_digits), 'Number of itteration steps:', self.count, '\n')
            return

    def Calculate_new_del(self):
        b = self.Hamiltonian.MF_params
        occupied_eigenvectors = self.Eigenvectors[self.indices_array[0], :, self.indices_array[-1]]
        a = self.Hamiltonian.Consistency(occupied_eigenvectors)
        return a, b

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
        elif self.method == 'decaying':
            beta = 1/(1 + self.beta)**self.count
        elif self.method == 'exponential':
            beta = np.exp(-self.count*self.beta)
        elif self.method == 'decaying sinusoidal':
            beta = np.exp(-self.count/self.Itteration_limit)*np.abs(np.sin(2*np.pi*self.count/self.Itteration_limit))
        else:
            print('Error: Itteration Method not found')

        if self.count == 0:
            beta = 0
        elif self.count >= 0.7*self.Itteration_limit and self.count % int(self.Itteration_limit/10) == 0:
            beta = 0.5
        elif self.count == int(0.9*self.Itteration_limit):
            beta = 1

        return (1 - beta)*b + beta*a, beta

    def Itteration_Step(self, verbose):
        #   Calculate Dynamic Variables
        self.Hamiltonian.update_variables()
        # Solve Matrix Across all momenta
        self.Energies, self.Eigenvectors = self.Hamiltonian.Mat_q_calc()
        # Find Indices of all required lowest energies
        self.Find_filling_lowest_energies()
        # Calculate Mean Field Parameters with lowest energies
        New_MFP, previous_MFP = self.Calculate_new_del()
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

    def Calculate_Total_E(self):
        self.occupied_energies = self.Energies[self.indices_array]
        self.Fermi_Energy = np.max(self.occupied_energies)
        total_occ = np.sum(self.occupied_energies)
        return self.Hamiltonian.Calculate_Energy(total_occ)

    def MIT_determination(self, binning='fd'):
        self.Energies = np.sort(self.Energies)

        hist, bins = np.histogram(self.Energies, bins=binning)
        a = np.digitize(self.Fermi_Energy, bins)
        if a < len(hist):
            if hist[a] > 0:
                self.Conductor = True
            else:
                self.Conductor = False
        else:
            self.Conductor = False

    def bandwidth_calculation(self, binning='fd'):
        hist, bins = np.histogram(self.Energies, bins=binning)
        a = np.digitize(self.Fermi_Energy, bins)
        E_dist = np.mean(np.diff(bins))

        bandwidths = []
        count = 0
        hist = np.sign(hist)
        for i, v in enumerate(hist):
            if v == 1:
                count += 1
            elif v == -1:
                count += -1
            if i < len(hist)-1:
                if v*hist[i+1] == 0:
                    bandwidths.append(count)
                    count = 0
            if i == len(hist)-1:
                bandwidths.append(count)

        bandwidths_per_E = []
        for v in bandwidths:
            if v != 0:
                bandwidths_per_E = bandwidths_per_E + v*[v]
            elif v == 0:
                bandwidths_per_E = bandwidths_per_E + [0]

        bandwidths_per_E = E_dist*np.array(bandwidths_per_E)

        self.Fermi_bandwidth = bandwidths_per_E[a-1]
        return self.Fermi_bandwidth

    def Itterate(self, verbose=True, save_seq=False, order=None):
        self.count = 0
        self.converged = True

        c = self.Hamiltonian.MF_params
        if verbose:
            self.Print_step(c, method='Initial')

        if self.save_seq:
            self.sol_seq = []
            self.beta_seq = []

        a, b = self.Itteration_Step(verbose)

        while LA.norm(a-c, ord=order) > self.tol:
            c = b
            a, b = self.Itteration_Step(verbose)
            if self.count >= self.Itteration_limit:
                self.converged = False
                break

        if self.save_seq:
            self.sol_seq = np.vstack(self.sol_seq)
            self.beta_seq = np.vstack(self.beta_seq)

        if verbose:
            self.Print_step(a, method='Final')

        if self.converged:
            self.Final_Total_Energy = self.Calculate_Total_E()
        else:
            self.Final_Total_Energy = np.inf

        self.MIT_determination()
