import numpy as np

PDE_dim = 2
PDE_vars = ['x', 'y']
PDE_scale = {
    'x': (0, 1),
    'y': (0, 1)
}
PDE_analytic_solution = True
PDE_description = 'Uxx + Uyy + U * Uy= PDE_f1(x, y)'
PDE_initial_condition = []
PDE_boundary_condition = ['PDE_c1(y, x=0)', 'PDE_c2(y, x=1)', 'PDE_c3(x, y=0)', 'PDE_c4_Uy(x, y=1)']


def PDE_f1(x, y):
    return np.sin(np.pi * x) * (2 - np.power(np.pi, 2) * np.power(y, 2) + 2 * np.power(y, 3) * np.sin(np.pi * x))


def PDE_c1(y, x=0):
    return y - y


def PDE_c2(y, x=1):
    return y - y


def PDE_c3(x, y=0):
    return y - y


def PDE_c4_Uy(x, y=1):
    return 2 * np.sin(np.pi * x)


def PDE_u(x, y):
    return np.power(y, 2) * np.sin(np.pi * x)
