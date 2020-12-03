import os
import itertools
import numpy as np


def spin_interpreter(mfps, rounding=1):
    b1 = np.array([1, 1])
    b2 = np.array([1, -1])

    spin = mfps[1]*b1 + mfps[3]*b2

    for i, v in enumerate(spin):
        if 0.1 < v < 0.5:
            spin[i] += 0.5
        elif -0.1 > v > -0.5:
            spin[i] -= 0.5

    spin = np.rint(spin*rounding)

    return spin


def orbit_interpreter(mfps, rounding=1):
    b1 = np.array([1, 1])
    b2 = np.array([1, -1])

    orbit = mfps[2]*b1 + mfps[4]*b2

    for i, v in enumerate(orbit):
        if 0.1 < v < 0.5:
            orbit[i] += 0.5
        elif -0.1 > v > -0.5:
            orbit[i] -= 0.5

    orbit = np.rint(orbit*rounding)
    return orbit


def symetries(phase):
    spin, orbit = phase[:2], phase[2:]
    # spin
    if np.product(np.sign(spin)) == 1:
        spin = np.abs(spin)
    if np.product(np.sign(spin)) == -1 and np.sign(spin[0]) == -1:
        spin = np.roll(spin, 1)
    if spin[0] == 0:
        spin = np.roll(spin, 1)
    if 0 in spin:
        spin = np.abs(spin)
    # orbit
    if orbit[0] == 0:
        orbit = np.roll(orbit, 1)
    if np.product(np.sign(orbit)) == -1 and np.sign(orbit[0]) == -1:
        orbit = np.roll(orbit, 1)
    phase[:2], phase[2:] = spin, orbit
    return phase


def fullphase(mfp):
    phase = np.zeros(5)
    phase[0] = mfp[0]
    spin = spin_interpreter(mfp)
    orbit = orbit_interpreter(mfp)
    phase[1:3], phase[3:] = spin, orbit
    phase[1:] = symetries(phase[1:])
    return phase


def array_interpreter(MFPs):
    Full_phase = np.zeros((MFPs.shape[0], MFPs.shape[1], 5))

    for v in itertools.product(np.arange(MFPs.shape[0]), np.arange(MFPs.shape[1])):
        Full_phase[v] = fullphase(MFPs[v])
    return Full_phase


def unique_states(state_array):
    states = state_array.reshape(-1, state_array.shape[-1])
    states = np.unique(states, axis=0)
    return list(states)


def vec_to_int(x):
    x = x + 3
    x = x.astype(int)
    ints = ''
    for i in range(len(x)):
        ints = ints + str(x[i])
    ints = int(ints)
    return ints


def arr_to_int(MFPs):
    Full_phase = np.zeros((MFPs.shape[0], MFPs.shape[1]))
    for v in itertools.product(np.arange(MFPs.shape[0]), np.arange(MFPs.shape[1])):
        state = vec_to_int(np.array(MFPs[v]))
        state = state_to_pos[state]
        Full_phase[v] = state
    return Full_phase.astype(int)


Spin_Dict = {-2: r' \Downarrow', -1: r' \downarrow', 0: r' 0', 1: r' \uparrow', 2: r' \Uparrow'}
Orbit_Dict = {-2: r' \bar{Z}', -1: r' \bar{z}', 0: r' 0', 1: r' z', 2: r' Z'}

All_states_pre_sym = [np.array([i, j, k, l]) for i, j, k, l in itertools.product(np.arange(-2, 3), repeat=4)]
All_states_post_sym = np.unique([vec_to_int(symetries(state)) for state in All_states_pre_sym], axis=0)


state_to_pos = {state: i for i, state in enumerate(All_states_post_sym)}

indices = np.arange(len(state_to_pos))
np.random.seed(42)
np.random.shuffle(indices)
state_to_pos_rand = {All_states_post_sym[v]: i for i, v in enumerate(indices)}
# state_to_pos = state_to_pos_rand

state_to_label = {
    vec_to_int(np.array([i, j, k, l])): r'$'+Spin_Dict[i]+Spin_Dict[j] + ', ' + Orbit_Dict[k]+Orbit_Dict[l]+'$'
    for i, j, k, l in itertools.product(np.arange(-2, 3), repeat=4)
}

pos_to_label = {
    state_to_pos[state]: state_to_label[state] for state in All_states_post_sym
}


def phase_to_label(phase):
    out = vec_to_int(symetries(phase))
    out = state_to_pos[out]
    print('label: ', out)
    print('checking: ', pos_to_label[out])


# phase_to_label(np.array([1, -1, -1, -1]))
# print(vec_to_int(symetries(np.array([1, -1, -1, -1]))))
# print(state_to_pos[4222])
# print(pos_to_label[38])