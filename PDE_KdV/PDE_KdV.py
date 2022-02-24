import numpy as np
import scipy.io as reader

PDE_datafile_sin = 'KdV_sine.mat'
PDE_datafile_cos = 'KdV_cos.mat'
PDE_dim = 2
PDE_vars = ['x', 't']
PDE_scale = {
    'x': (-20, 20),
    't': (0, 40)
}
PDE_analytic_solution = False
PDE_description = 'Ut = - U * Ux - Uxxx'
PDE_initial_condition_sin = ['PDE_ic1_sin(x, t=0)']
PDE_initial_condition_cos = ['PDE_ic1_cos(x, t=0)']
PDE_boundary_condition = ['PDE_bc1 : U(t, x=-20) = U(t, x=20)', 'PDE_bc2 : Ux(t, x=-20) = Ux(t, x=20)','PDE_bc3 : Uxx(t, x=-20) = Uxx(t, x=20)']

def PDE_ic1_sin(x, t=0):
    return - np.sin(np.pi * x/20)

def PDE_ic1_cos(x, t=0):
    return np.cos( - np.pi * x/20)


def PDE_get_data_sin():
    data = reader.loadmat(PDE_datafile_sin)
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    X ,T= np.meshgrid(x, t)
    return {
        'data': data,
        'x': x,
        'x_dim': len(x),
        't': t,
        't_dim': len(t),
        'solution_type': 'X T U',
        'solution': [X, T, Exact]
    }

def PDE_get_data_cos():
    data = reader.loadmat(PDE_datafile_cos)
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    X,T = np.meshgrid(x, t)
    return {
        'data': data,
        'x': x,
        'x_dim': len(x),
        't': t,
        't_dim': len(t),
        'solution_type': 'X T U',
        'solution': [X, T, Exact]
    }
