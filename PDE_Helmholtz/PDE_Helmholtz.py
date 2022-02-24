import numpy as np

PDE_dim = 2
PDE_vars = ['x', 'y']
PDE_scale = {
    'x': (-1, 1),
    'y': (-1, 1)
}
PDE_analytic_solution = True
PDE_description = 'Uxx + Uyy + k^2 * U= PDE_f1(x, y) param: k a1 a2'
PDE_initial_condition = []
PDE_boundary_condition = ['4 PDE_c1(x,y)']

k=1
a1=1
a2=4

def PDE_f1(x, y):
    return -np.power(a1*np.pi,2)*PDE_u(x,y)-np.power(a2*np.pi,2)*PDE_u(x,y)+np.power(k,2)*PDE_u(x,y)



def PDE_c1(x,y):
    return PDE_u(x,y)


def PDE_u(x, y):
    return  np.sin(a1*np.pi * x)*np.sin(a2*np.pi * y)
