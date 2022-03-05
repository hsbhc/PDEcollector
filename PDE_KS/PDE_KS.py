import os

import numpy as np
import scipy.io as reader
abs_dir= os.path.dirname(__file__) + '/'
PDE_datafile = abs_dir + 'KS.mat'
PDE_dim = 2
PDE_vars = ['x', 't']
PDE_scale = {
    'x': (-10, 10),
    't': (0, 50)
}
PDE_analytic_solution = False
PDE_description = 'Ut = - U * Ux - Uxx -Uxxxx'
PDE_initial_condition= ['PDE_ic1(x, t=0)']
PDE_boundary_condition = ['PDE_bc1 : U(t, x=-10) = U(t, x=10)', 'PDE_bc2 : Ux(t, x=-10) = Ux(t, x=10)',
                          'PDE_bc3 : Uxx(t, x=-10) = Uxx(t, x=10)','PDE_bc4 : Uxxx(t, x=-10) = Uxxx(t, x=10)']

def PDE_ic1(x, t=0):
    return - np.sin(np.pi * x/10)


def PDE_get_data():
    data = reader.loadmat(PDE_datafile)
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

data=PDE_get_data()['solution']
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)