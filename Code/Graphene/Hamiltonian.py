import numpy as np
from numpy import linalg as LA


class Hamiltonian:
    """
    """
    def __init__(self, Model_params):
        # initiates Model parameters
        self.t_1 = 1
        self.t_2 = 0.15
        self.t_4 = 0.05
        self.stress = 0

        for key, value in Model_params.items():
            setattr(self, key, value)

        self.k_points = np.array([[0, 0],
                                  2*np.pi/3*np.array([1, 1/np.sqrt(3)]),
                                  2*np.pi/3*np.array([1, 0]),
                                  [0, 0]])

        self.k_labels = [r'$\Gamma$', 'K', 'M', r'$\Gamma$']

        self.mat_dim = 4

    def func_tzz(self, qx, qy):
        return -self.t_1*np.exp(-1j*qx)*(
                1 + 2*np.exp(3*qx/2*1j)*np.cos(np.sqrt(3)*qy/2)
                )

    def static_variables(self):
        # Static variables, these never change, may depend on momentum indices
        # self.Q gets updated outside
        qx, qy = self.Q
        self.tzz = self.func_tzz(qx, qy)

    def matrices(self):
        zeros = np.zeros(self.Q[0].shape)
        matrices = np.array([
                [zeros, self.tzz],
                [np.conjugate(self.tzz), zeros]
                ])
        matrices = np.moveaxis(matrices, [0, 1], [-2, -1])
        w, v = LA.eigh(matrices)
        return w, np.moveaxis(v, -1, -2)
