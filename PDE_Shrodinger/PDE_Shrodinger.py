import os

import numpy as np
import scipy.io as reader

abs_dir= os.path.dirname(__file__) + '/'
PDE_datafile = abs_dir + 'NLS.mat'
PDE_dim = 2
PDE_vars = ['x', 't']
PDE_scale = {
    'x': (-5, 5),
    't': (0, np.pi / 2)
}
PDE_analytic_solution = False
PDE_description = 'i * Ut + 0.5 * Uxx + |U|^2 * U = 0'
PDE_initial_condition = ['PDE_ic1(x, t=0)']
PDE_boundary_condition = ['PDE_bc1 : U(t, x=-5) = U(t, x=5)', 'PDE_bc2 : Ux(t, x=-5) = Ux(t, x=5)']


def PDE_ic1(x, t=0):
    return 2 * (1 / np.cosh(x))


def PDE_bc1():
    return 'U(t, x=-5) = U(t, x=5)'


def PDE_bc2():
    return 'Ux(t, x=-5) = Ux(t, x=5)'


def PDE_get_data():
    data = reader.loadmat(PDE_datafile)
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu'].T
    Exact_a = np.real(Exact)
    Exact_b = np.imag(Exact)
    Exact_h = np.sqrt(Exact_a ** 2 + Exact_b ** 2)
    X, T = np.meshgrid(x, t)

    return {
        'data': data,
        'x': x,
        'x_dim': len(x),
        't': t,
        't_dim': len(t),
        'solution_type': 'U = a + b * i , X T |U| a b ',
        'solution': [X, T, Exact_h, Exact_a, Exact_b]
    }
