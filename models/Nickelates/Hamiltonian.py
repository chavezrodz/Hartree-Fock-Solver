import numpy as np
from numpy import linalg as LA


class Hamiltonian:
    """
    Contains: Matrix structure, elements,
    consistency equations, total energy equation
    and both static and dynamic parameters

    Model_Params must be a dictionary and at least contain:
    N_dim
    N_cells
    Filling
    mat_dim

    MF_Params must be a 1D np.array

    The class must contain the methods:
    update_variables
    Mat_q_calc
    Consistency
    Calculate_Energy

    All iterations done in HFA solver.
    """
    def __init__(self, Model_params={}, MF_params=np.array([0, 0, 0, 0, 0])):

        # MFPs Names
        self.Dict = {0: 'Charge Modulation',
                     1: 'Ferromagnetism',
                     2: 'Orbital Disproportionation',
                     3: 'Anti Ferromagnetism',
                     4: 'Anti Ferroorbital'}
        # MFP Symbols
        self.dict_symbol = {0: r'$\delta$',
                     1: r'$S_{FM}$',
                     2: r'$O_{FM}$',
                     3: r'$S_{AF}$',
                     4: r'$O_{AF}$'}

        # K_patn for bandstructure
        self.k_points = np.array([[0, 0, 0],
                                  [np.pi, 0, 0],
                                  [np.pi, np.pi/2, 0],
                                  [0, 0, 0]])
        self.k_labels = ['M', r'$\Gamma$', 'X', 'M']

        # initiates Model parameters
        self.mat_dim = 8
        self.BZ_rot = 1
        self.b = 0

        self.k_res = 100
        self.n_dim = 2

        self.Filling = 0.25
        self.stress = 0
        self.Delta_CT = 0
        self.eps = 0

        self.t_1 = 1
        self.t_2 = 0.15
        self.t_4 = 0.05
        self.U = 0
        self.J = 0

        for key, value in Model_params.items():
            setattr(self, key, value)

        # initiates Mean field parameters
        self.MF_params = MF_params

        # define some numbers
        N = np.power(self.k_res, self.n_dim) * self.mat_dim
        N_c = 8
        N_k = N / N_c
        self.N_ni = 2 * N_k
        N_s = N_k * 4

        # projectors for DOSs
        Id = np.identity(8)
        z2_projectors = Id[[0, 1, 4, 5]]
        x2my2_projectors = Id[[2, 3, 6, 7]]

        hom1_up = Id[[0, 2]]
        hom2_up = Id[[1, 3]]
        hom1_down = Id[[4, 6]]
        hom2_down = Id[[5, 7]]

        site1_up = (hom1_up+hom2_up)/np.sqrt(2.)
        site1_down = (hom1_down+hom2_down)/np.sqrt(2.)
        site2_up = (hom1_up-hom2_up)/np.sqrt(2.)
        site2_down = (hom1_down-hom2_down)/np.sqrt(2.)

        self.proj_configs = [
               {"name": 'Orbit',
                "title": '',
                "proj_1": z2_projectors,
                "proj_2": x2my2_projectors,
                "normalizer": self.N_ni,
                "/dE": True,
                "label_1": r'$3z^2-r^2$',
                "label_2": r'$x^2-y^2$'
                },

               {"name": 'Site 1',
                "title": 'Site 1',
                "proj_1": site1_up,
                "proj_2": site1_down,
                "normalizer": 0.5*self.N_ni,
                "/dE": True,
                "label_1": r'$\uparrow$',
                "label_2": r'$\downarrow$'
                },

               {"name": 'Site 2',
                "title": 'Site 2',
                "proj_1": site2_up,
                "proj_2": site2_down,
                "normalizer": 0.5*self.N_ni,
                "/dE": True,
                "label_1": r'$\uparrow$',
                "label_2": r'$\downarrow$'
                },

               # {"name": 'spins',
               #  "title": '',
               #  "proj_1": hom1_up + hom2_up,
               #  "proj_2": hom1_down + hom2_down,
               #  "normalizer": self.N_ni,
               #  "/dE": True,
               #  "label_1": r'$\uparrow$',
               #  "label_2": r'$\downarrow$'
               #  }
        ]

    def func_tzz(self, Q):
        qx, qy, qz = Q
        B = self.b
        return -2*self.t_1*(B*np.cos(qz) + 1/4*(np.cos(qx) + np.cos(qy)))\
            - 2*self.t_4*(B*np.cos(2*qz) + 1/4*(np.cos(2*qx) + np.cos(2*qy)))\
            - 2*self.t_2*(np.cos(qx)*np.cos(qy) - 2*B*np.cos(qz)*(np.cos(qy) + np.cos(qx)))

    def func_tz_bz_b(self, Q):
        qx, qy, qz = Q
        return -3/2*self.t_1*(np.cos(qx) + np.cos(qy))\
            - 3/2*self.t_4*(np.cos(2*qx) + np.cos(2*qy))\
            + 6*self.t_2*np.cos(qx)*np.cos(qy)

    def func_tzz_b(self, Q):
        qx, qy, qz = Q
        B = self.b
        return np.sqrt(3)/2*self.t_1*(np.cos(qx) - np.cos(qy))\
            + np.sqrt(3)/2*self.t_4*(np.cos(2*qx) - np.cos(2*qy))\
            - 2*np.sqrt(3)*self.t_2*B*np.cos(qz)*(np.cos(qx) - np.cos(qy))

    def static_variables(self):
        # Static variables, these never change, may depend on momentum indices

        # Strain Effect
        decay = 1
        self.f = 4*self.Filling
        self.t_1 = self.t_1*np.exp(-decay*self.stress)
        self.t_2 = self.t_2*np.exp(-decay*np.sqrt(2)*self.stress)
        self.t_4 = self.t_4*np.exp(-decay*2*self.stress)

        self.N_cells = np.power(self.k_res, self.n_dim)

        # self.Q gets updated outside
        qc = np.pi
        Q = self.Q
        Qc = self.Q + qc

        self.tzz = self.func_tzz(Q)
        self.tzz_c = self.func_tzz(Qc)
        self.tz_bz_b = self.func_tz_bz_b(Q)
        self.tz_bz_b_c = self.func_tz_bz_b(Qc)
        self.tzz_b = self.func_tzz_b(Q)
        self.tzz_b_c = self.func_tzz_b(Qc)

    def dynamic_variables(self):
        """
        Calculate dynamic variables
        These depend on MFP, not on momentum
        """
        self.U_0 = (self.U + self.J)/2
        self.U_bar = (3*self.U - 5*self.J)/4
        self.J_bar = (-self.U + 5*self.J)/2

        # Distortion
        alpha = 1
        beta = 27/4*alpha*self.MF_params[0]**2

        if np.abs(self.MF_params[0]) < 1e-12:
            self.u = 0
        else:
            self.u = 3*self.MF_params[0] / (2*np.cbrt(beta)) * (np.cbrt(1 + np.sqrt(1 + 1/beta)) + np.cbrt(1 - np.sqrt(1 + 1/beta)))

        sigma = 1
        self.a0p = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] + self.J_bar*self.MF_params[4]
        self.a1p = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] - self.J_bar*self.MF_params[4]

        sigma = -1
        self.a0m = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] + self.J_bar*self.MF_params[4]
        self.a1m = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] - self.J_bar*self.MF_params[4]

    def submatrix(self, sigma):
        # Call matrix elements

        if sigma == 1:
            a0 = np.full(self.Q[0].shape, self.a0p)
            a1 = np.full(self.Q[0].shape, self.a0p)
        elif sigma == -1:
            a0 = np.full(self.Q[0].shape, self.a0m)
            a1 = np.full(self.Q[0].shape, self.a0m)

        zeros = np.zeros(self.Q[0].shape)
        b = self.tzz_b
        c = self.tzz_b_c

        d0 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] + self.J_bar*self.MF_params[2] + self.tzz + self.Delta_CT/2
        d1 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] + self.J_bar*self.MF_params[2] + self.tzz_c + self.Delta_CT/2
        d2 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] - self.J_bar*self.MF_params[2] + self.tz_bz_b - self.Delta_CT/2
        d3 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] - self.J_bar*self.MF_params[2] + self.tz_bz_b_c - self.Delta_CT/2

        # Declare sub-block
        sub = np.array([
                [d0, a0, b, zeros],
                [a0, d1, zeros, c],
                [b, zeros, d2, a1],
                [zeros, c, a1, d3]])

        return np.moveaxis(sub, [0, 1], [-2, -1])

    def matrices(self):
        self.dynamic_variables()

        sub_1 = self.submatrix(1)
        sub_2 = self.submatrix(-1)

        zeros = np.zeros(sub_1.shape)
        matrices = np.block([
                [sub_1, zeros],
                [zeros, sub_2]
                ])
        w, v = LA.eigh(matrices)
        return w, np.moveaxis(v, -1, -2)

    def Consistency(self, v):
        """
        Consistency Equations, keeps order of MFPs
        v: occupied eigenvectors, shape = (N_occ,dims)
        """
        a = 0.5*(np.conj(v[:, 0])*v[:, 1] + np.conj(v[:, 2])*v[:, 3] +
                 np.conj(v[:, 4])*v[:, 5] + np.conj(v[:, 6])*v[:, 7] +
                 np.conj(v[:, 1])*v[:, 0] + np.conj(v[:, 3])*v[:, 2] +
                 np.conj(v[:, 5])*v[:, 4] + np.conj(v[:, 7])*v[:, 6])

        b = 0.5*(np.abs(v[:, 0])**2 + np.abs(v[:, 1])**2 +
                 np.abs(v[:, 2])**2 + np.abs(v[:, 3])**2 -
                 np.abs(v[:, 4])**2 - np.abs(v[:, 5])**2 -
                 np.abs(v[:, 6])**2 - np.abs(v[:, 7])**2)

        c = 0.5*(np.abs(v[:, 0])**2 + np.abs(v[:, 1])**2 -
                 np.abs(v[:, 2])**2 - np.abs(v[:, 3])**2 +
                 np.abs(v[:, 4])**2 + np.abs(v[:, 5])**2 -
                 np.abs(v[:, 6])**2 - np.abs(v[:, 7])**2)

        d = 0.5*(np.conj(v[:, 0])*v[:, 1] + np.conj(v[:, 2])*v[:, 3] -
                 np.conj(v[:, 4])*v[:, 5] - np.conj(v[:, 6])*v[:, 7] +
                 np.conj(v[:, 1])*v[:, 0] + np.conj(v[:, 3])*v[:, 2] -
                 np.conj(v[:, 5])*v[:, 4] - np.conj(v[:, 7])*v[:, 6])

        e = 0.5*(np.conj(v[:, 0])*v[:, 1] - np.conj(v[:, 2])*v[:, 3] +
                 np.conj(v[:, 4])*v[:, 5] - np.conj(v[:, 6])*v[:, 7] +
                 np.conj(v[:, 1])*v[:, 0] - np.conj(v[:, 3])*v[:, 2] +
                 np.conj(v[:, 5])*v[:, 4] - np.conj(v[:, 7])*v[:, 6])

        sub_params = np.real([a, b, c, d, e])/self.N_cells

        return np.sum(sub_params, axis=1)

    def Calculate_Energy(self, E_occ):
        E = E_occ/self.N_cells
        E += 2*self.eps*(self.u**2/2 + self.u**4/4)
        E -= (
            self.U_bar/2*(self.f**2+self.MF_params[0]**2)
            - self.U_0*(self.MF_params[1]**2 + self.MF_params[3]**2)
            + self.J_bar*(self.MF_params[2]**2 + self.MF_params[4]**2)
             )
        return E
