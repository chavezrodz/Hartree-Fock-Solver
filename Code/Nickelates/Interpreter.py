import itertools
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib


def spin_interpreter(mfps, rounding=1):
    shape = (*mfps.shape[:-1], 2)
    b1 = np.ones(shape)
    b2 = np.ones(shape)
    b2[..., 1] = -1

    b1 = np.einsum('ij,ijk->ijk', mfps[..., 1], b1)
    b2 = np.einsum('ij,ijk->ijk', mfps[..., 3], b2)
    spin = b1 + b2

    spin[(0.1 < spin) & (spin < 0.5)] += 0.5
    spin[(-0.5 < spin) & (spin < -0.1)] -= 0.5

    spin = np.rint(spin*rounding)

    return spin


def orbit_interpreter(mfps, rounding=1):
    shape = (*mfps.shape[:-1], 2)
    b1 = np.ones(shape)
    b2 = np.ones(shape)
    b2[..., 1] = -1

    b1 = np.einsum('ij,ijk->ijk', mfps[..., 2], b1)
    b2 = np.einsum('ij,ijk->ijk', mfps[..., 4], b2)
    orbit = b1 + b2

    orbit[(0.1 < orbit) & (orbit < 0.5)] += 0.5
    orbit[(-0.5 < orbit) & (orbit < -0.1)] -= 0.5

    orbit = np.rint(orbit*rounding)
    return orbit


def symetries(phase):
    spin, orbit = phase[:2], phase[2:]


    idx = (0 in spin) or (np.product(np.sign(spin), axis=-1) == 1)
    spin[idx] = np.abs(spin)

    idx = (spin[0] == 0) & (orbit[0] == 0)
    orbit[idx] = np.roll(orbit, 1, axis=-1)
    spin[idx] = np.roll(spin, 1, axis=-1)

    idx = (orbit[0] == orbit[1]) & (spin[0] < spin[1])
    spin[idx] = np.roll(spin, 1, axis=-1)

    idx = (spin[0] == spin[1]) & (orbit[0] < orbit[1])
    orbit[idx] = np.roll(orbit, 1, axis=-1)

    idx = (spin[0] < spin[1]) & (orbit[0] < orbit[1])
    spin[idx] = np.roll(spin, 1, axis=-1)
    orbit[idx] = np.roll(orbit, 1, axis=-1)

    spin = np.sign(spin)
    orbit = np.sign(orbit)

    phase[:2], phase[2:] = spin, orbit

    return phase


def array_interpreter(mfp):
    phase = np.zeros(mfp.shape)
    CM = mfp[..., 0]
    OS = mfp[..., 2]
    spin = spin_interpreter(mfp)
    orbit = orbit_interpreter(mfp)
    phase[..., 0], phase[..., 1:3], phase[..., 3:] = CM, spin, orbit
    for v in itertools.product(np.arange(mfp.shape[0]), np.arange(mfp.shape[1])):
        phase[v][1:] = symetries(phase[v][1:])
        phase[v][1] = vec_to_int(phase[v][1:])
        phase[v][1] = state_to_pos[phase[v][1]]
    phase[..., 2] = OS
    return phase[..., :3]


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


Spin_Dict = {-2: r' \Downarrow', -1: r' \downarrow', 0: r' 0', 1: r' \uparrow', 2: r' \Uparrow'}
Orbit_Dict = {-2: r' \bar{Z}', -1: r' \bar{z}', 0: r' 0', 1: r' z', 2: r' Z'}
All_states_pre_sym = [np.array([i, j, k, l]) for i, j, k, l in itertools.product(np.arange(-2, 3), repeat=4)]
All_states_post_sym = np.unique([vec_to_int(symetries(state)) for state in All_states_pre_sym], axis=0)

N_possible_states = len(All_states_post_sym)
state_to_pos = {state: i for i, state in enumerate(All_states_post_sym)}

# indices = np.arange(len(state_to_pos))
# np.random.seed(42)
# np.random.shuffle(indices)
# state_to_pos_rand = {All_states_post_sym[v]: i for i, v in enumerate(indices)}
# state_to_pos = state_to_pos_rand

state_to_label = {
    vec_to_int(np.array([i, j, k, l])): r'$'+Spin_Dict[i]+Spin_Dict[j] + ', ' + Orbit_Dict[k]+Orbit_Dict[l]+'$'
    for i, j, k, l in itertools.product(np.arange(-2, 3), repeat=4)
}

pos_to_label = {
    state_to_pos[state]: state_to_label[state] for state in All_states_post_sym
}
# print(pos_to_label)


def phase_to_label(phase):
    out = vec_to_int(symetries(phase))
    phase_integer = state_to_pos[out]
    phase_string = pos_to_label[phase_integer]
    return phase_string, phase_integer


states_of_interest = np.array([
    [0,  0, -1, -2],
    [0,  0, -1, -1],
    [0,  0, -1,  0],
    [0,  0,  0,  0],
    [0,  0,  0, -1],
    [1, -1, -1, -1],
    [1, -1,  1,  1],
    [1,  0, -1, -1],
    [1,  0, -1,  0],
    [1,  1, -1, -1],
    [1,  1,  0, -1],
    [1,  1,  0,  0],
    [1,  1,  1,  1],
    [2,  0, -1,  0],
    [2,  0,  0,  0],
    [2,  1, -1, -1],
    [2,  1, -1,  0],
    [2,  1,  0, -1],
    [2,  1,  0,  0]
    ])


colors_of_interest = [
    'tab:red',
    'tab:blue',
    'tab:purple',
    'red',
    'yellow',
    'green',
    'brown',
    'darkgrey',
    'tab:green',
    'tab:gray',
    'tab:olive',
    'tab:orange',
    'blue',
    'purple',
    'tab:pink',
    'navajowhite',
    'tab:brown',
    'lime',
    'tab:cyan'
]

assert len(states_of_interest) <= len(colors_of_interest)

strings_of_interest = [phase_to_label(v)[0] for v in states_of_interest]
integers_of_interest = [phase_to_label(v)[1] for v in states_of_interest]


# Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
col_dict = dict(zip(
    integers_of_interest, colors_of_interest[:len(integers_of_interest)]
    ))

# We create a colormar from our list of colors
custom_cmap = ListedColormap([col_dict[x] for x in col_dict.keys()])

# Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or using another dict maybe could help.
labels = np.array(strings_of_interest)
len_lab = len(labels)

# prepare normalizer
#  Prepare bins for the normalizer
norm_bins = np.sort([*col_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

# Make normalizer and formatter
custom_norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)


print(f'Total Possible States: {N_possible_states}')
# print(phase_to_label(np.array([1, -1, -1, -1])))
# print(vec_to_int(symetries(np.array([1, -1, -1, -1]))))
# print(state_to_pos[4222])
# print(pos_to_label[45])