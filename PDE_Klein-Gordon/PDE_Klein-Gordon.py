import numpy as np

PDE_dim = 2
PDE_vars = ['x', 't']
PDE_scale = {
    'x': (0, 1),
    't': (0, 1)
}
PDE_analytic_solution = True
PDE_description = 'Utt + a * Uxx + b * U + c *U^k= PDE_f1(x, t) param: k a b c'
PDE_initial_condition = ['PDE_i1(x, t=0)', 'PDE_i2_Ut(x, t=0)']
PDE_boundary_condition = ['PDE_c1(t, x=0)', 'PDE_c2(t, x=1)']

k = 3
a = -1
b = 0
c = 1


def PDE_i1(x, t=0):
    return PDE_u(x, t)


def PDE_i2_Ut(x, t=0):
    return x - x


def U_tt(x, t):
    return - 25 * np.pi ** 2 * x * np.cos(5 * np.pi * t) + 6 * t * x ** 3


def U_xx(x, t):
    return 6 * x * t ** 3


def PDE_f1(x, t):
    return U_tt(x, t) + a * U_xx(x, t) + b * PDE_u(x, t) + c * PDE_u(x, t) ** k


def PDE_c1(t, x=0):
    return PDE_u(x, t)


def PDE_c2(t, x=1):
    return PDE_u(x, t)


def PDE_u(x, t):
    return x * np.cos(5 * np.pi * t) + np.power(x * t, 3)
