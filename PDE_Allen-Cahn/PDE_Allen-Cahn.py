import numpy as np
import scipy.io as reader

PDE_datafile = 'AC.mat'
PDE_dim = 2
PDE_vars = ['x', 't']
PDE_scale = {
    'x': (-1, 1),
    't': (0, 1)
}
PDE_analytic_solution = False
PDE_description = 'Ut - 0.0001 * Uxx + 5 * U^3 - 5 * U = 0'
PDE_initial_condition = ['PDE_ic1(x, t=0)']
PDE_boundary_condition = ['PDE_bc1 : U(t, x=-1) = U(t, x=1)', 'PDE_bc2 : Ux(t, x=-1) = Ux(t, x=1)']


def PDE_ic1(x, t=0):
    return np.power(x,2)*np.cos(np.pi*x)


def PDE_bc1():
    return 'U(t, x=-1) = U(t, x=1)'


def PDE_bc2():
    return 'Ux(t, x=-1) = Ux(t, x=1)'


def PDE_get_data():
    data = reader.loadmat(PDE_datafile)
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['uu']).T

    X, T = np.meshgrid(x, t)

    return {
        'data': data,
        'x': x,
        'x_dim': len(x),
        't': t,
        't_dim': len(t),
        'solution_type': 'X T U',
        'solution': [X, T, Exact]
    }

