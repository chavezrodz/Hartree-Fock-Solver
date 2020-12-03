import numpy as np
from numpy import linalg as LA


class Hamiltonian:
    """
    Contains: Matrix structure, elements, consistency equations, total energy equation
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

    All itterations done in HFA solver.
    """
    def __init__(self, Model_params, MF_params=np.array([0, 0, 0, 0, 0])):

        # initiates Mean field parameters
        self.MF_params = MF_params

        # initiates Model parameters
        self.N_shape = (50, 50)
        self.Filling = 0.25
        self.BZ_rot = 1
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

        # Strain Effect
        decay = 1
        self.f = 4*self.Filling
        self.t_1 = self.t_1*np.exp(-decay*self.stress)
        self.t_2 = self.t_2*np.exp(-decay*np.sqrt(2)*self.stress)
        self.t_4 = self.t_4*np.exp(-decay*self.stress)

        self.Dict = {0: 'Charge Modulation',
                     1: 'Ferromagnetism',
                     2: 'Orbital Disproportionation',
                     3: 'Anti Ferromagnetism',
                     4: 'Anti Ferroorbital'}

        self.mat_dim = 8
        self.N_dim = len(self.N_shape)
        self.N_cells = int(np.prod(self.N_shape))

        # Allowed Momentum Values
        self.Qv = np.mgrid[
            -np.pi:np.pi:(self.N_shape[0]*1j),
            -np.pi:np.pi:(self.N_shape[1]*1j)].reshape(self.N_dim, -1).T

        # Vectors Rotation by 45 degrees to re-create true BZ
        angle = -np.pi / 4.*self.BZ_rot
        rotate = np.array([
             [np.cos(angle), -np.sin(angle)],
             [np.sin(angle),  np.cos(angle)]])

        # scale to only allow up to pi/2 momentum values
        scaling = (1/np.sqrt(2.))*self.BZ_rot + 1.*(1. - self.BZ_rot)
        scale = np.array([
                 [scaling, 0],
                 [0, scaling]])

        self.Qv = np.array([np.dot(scale, np.dot(rotate, k)) for k in self.Qv])

        # Allowed Momentum Indices for itterator
        self.Qg = np.mgrid[
            0:self.N_shape[0],
            0:self.N_shape[1]].reshape(self.N_dim, -1).T

        self.Q = list(map(tuple, self.Qg))

        # Static variables, these never change, may depend on momentum indices
        qc = np.pi

        self.tzz = np.zeros(self.N_shape)
        self.tzz_c = np.zeros(self.N_shape)

        self.tz_bz_b = np.zeros(self.N_shape)
        self.tz_bz_b_c = np.zeros(self.N_shape)

        self.tzz_b = np.zeros(self.N_shape)
        self.tzz_b_c = np.zeros(self.N_shape)

        for i, q in enumerate(self.Q):
            qx, qy = self.Qv[i]

            self.tzz[q] = -self.t_1/2*(np.cos(qx) + np.cos(qy)) - self.t_4/2*(np.cos(2*qx) + np.cos(2*qy)) - 4*self.t_2*np.cos(qx)*np.cos(qy)
            self.tzz_c[q] = -self.t_1/2*(np.cos(qx+qc) + np.cos(qy+qc)) - self.t_4/2*(np.cos(2*(qx+qc)) + np.cos(2*(qy+qc))) - 4*self.t_2*np.cos(qx+qc)*np.cos(qy+qc)

            self.tz_bz_b[q] = -self.t_1*3/2*(np.cos(qx) + np.cos(qy)) - self.t_4*3/2*(np.cos(2*qx) + np.cos(2*qy)) - 12*self.t_2*np.cos(qx)*np.cos(qy)
            self.tz_bz_b_c[q] = -self.t_1*3/2*(np.cos(qx+qc) + np.cos(qy+qc)) - self.t_4*3/2*(np.cos(2*(qx+qc)) + np.cos(2*(qy+qc))) - 12*self.t_2*np.cos(qx+qc)*np.cos(qy+qc)

            self.tzz_b[q] = np.sqrt(3)/2*self.t_1*(np.cos(qx) - np.cos(qy)) + np.sqrt(3)/2*self.t_4*(np.cos(2*qx) - np.cos(2*qy))
            self.tzz_b_c[q] = np.sqrt(3)/2*self.t_1*(np.cos(qx+qc) - np.cos(qy+qc)) + np.sqrt(3)/2*self.t_4*(np.cos(2*(qx + qc)) - np.cos(2*(qy + qc)))

    def update_variables(self):
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

    def Mat_q_calc(self, q):
        """
        Declaration of the matrix to diagonalize, momentum dependent
        """

        # Call matrix elements
        sigma = 1

        a0 = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] + self.J_bar*self.MF_params[4]
        a1 = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] - self.J_bar*self.MF_params[4]
        b = self.tzz_b[q]
        c = self.tzz_b_c[q]

        d0 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] + self.J_bar*self.MF_params[2] + self.tzz[q] + self.Delta_CT/2
        d1 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] + self.J_bar*self.MF_params[2] + self.tzz_c[q] + self.Delta_CT/2
        d2 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] - self.J_bar*self.MF_params[2] + self.tz_bz_b[q] - self.Delta_CT/2
        d3 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] - self.J_bar*self.MF_params[2] + self.tz_bz_b_c[q] - self.Delta_CT/2

        # Declare sub-block
        sub_1 = np.array([
            [d0, a0, b, 0],
            [a0, d1, 0, c],
            [b, 0, d2, a1],
            [0, c, a1, d3]])

        # Call matrix elements
        sigma = -1

        a0 = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] + self.J_bar*self.MF_params[4]
        a1 = self.U_bar*self.MF_params[0] - 2*self.eps*self.u - sigma * self.U_0*self.MF_params[3] - self.J_bar*self.MF_params[4]
        b = self.tzz_b[q]
        c = self.tzz_b_c[q]

        d0 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] + self.J_bar*self.MF_params[2] + self.tzz[q] + self.Delta_CT/2
        d1 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] + self.J_bar*self.MF_params[2] + self.tzz_c[q] + self.Delta_CT/2
        d2 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] - self.J_bar*self.MF_params[2] + self.tz_bz_b[q] - self.Delta_CT/2
        d3 = self.U_bar*self.f - sigma*self.U_0*self.MF_params[1] - self.J_bar*self.MF_params[2] + self.tz_bz_b_c[q] - self.Delta_CT/2

        # Declare sub-block
        sub_2 = np.array([
            [d0, a0, b, 0],
            [a0, d1, 0, c],
            [b, 0, d2, a1],
            [0, c, a1, d3]])

        # Declare matrix
        mat = np.block([
            [sub_1, np.zeros((4, 4))],
            [np.zeros((4, 4)), sub_2]
            ])

        # Diagonalize Matrix
        w, v = LA.eig(mat)
        return w, v

    def Consistency(self, v):
        # Consistency Equations, keep order of MFP
        a = 0.5*(np.conj(v[0])*v[1] + np.conj(v[2])*v[3] + np.conj(v[4])*v[5] +
                 np.conj(v[6])*v[7] + np.conj(v[1])*v[0] + np.conj(v[3])*v[2] +
                 np.conj(v[5])*v[4] + np.conj(v[7])*v[6])/self.N_cells

        b = 0.5*(np.abs(v[0])**2 + np.abs(v[1])**2 + np.abs(v[2])**2 +
                 np.abs(v[3])**2 - np.abs(v[4])**2 - np.abs(v[5])**2 -
                 np.abs(v[6])**2 - np.abs(v[7])**2)/self.N_cells

        c = 0.5*(np.abs(v[0])**2 + np.abs(v[1])**2 - np.abs(v[2])**2 -
                 np.abs(v[3])**2 + np.abs(v[4])**2 + np.abs(v[5])**2 -
                 np.abs(v[6])**2 - np.abs(v[7])**2)/self.N_cells

        d = 0.5*(np.conj(v[0])*v[1] + np.conj(v[2])*v[3] - np.conj(v[4])*v[5] -
                 np.conj(v[6])*v[7] + np.conj(v[1])*v[0] + np.conj(v[3])*v[2] -
                 np.conj(v[5])*v[4] - np.conj(v[7])*v[6])/self.N_cells

        e = 0.5*(np.conj(v[0])*v[1] - np.conj(v[2])*v[3] + np.conj(v[4])*v[5] -
                 np.conj(v[6])*v[7] + np.conj(v[1])*v[0] - np.conj(v[3])*v[2] +
                 np.conj(v[5])*v[4] - np.conj(v[7])*v[6])/self.N_cells

        return a, b, c, d, e

    def Calculate_Energy(self, E_occ):
        E = E_occ/self.N_cells + 2*self.eps*(self.u**2/2 + self.u**4/4) - (self.U_bar/2*(self.f**2+self.MF_params[0]**2) - self.U_0*(self.MF_params[1]**2 + self.MF_params[3]**2) + self.J_bar*(self.MF_params[2]**2 + self.MF_params[4]**2))/self.N_cells
        return E
