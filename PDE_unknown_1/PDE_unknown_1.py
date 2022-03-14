import numpy as np
import torch

PDE_dim = 2
PDE_vars = ['x', 'y']
PDE_scale = {
    'x': (0, 1),
    'y': (0, 1)
}
PDE_analytic_solution = True
PDE_description = 'Uxx + Uyy = PDE_f1(x, y)'
PDE_initial_condition = []
PDE_boundary_condition = ['PDE_c1(y, x=0)', 'PDE_c2(y, x=1)', 'PDE_c3(x, y=0)', 'PDE_c4(x, y=1)']


def PDE_f1(x, y):

    return torch.exp(-x) * (x - 2 + torch.pow(y, 3) + 6 * y)


def PDE_c1(y, x=0):
    return np.power(y, 3)


def PDE_c2(y, x=1):
    return np.exp(-1) * (1 + np.power(y, 3))


def PDE_c3(x, y=0):
    return np.exp(-x) * x


def PDE_c4(x, y=1):
    return np.exp(-x) * (x + 1)


def PDE_u(x, y):
    return np.exp(-x) * (x + np.power(y, 3))
