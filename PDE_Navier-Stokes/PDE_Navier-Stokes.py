import numpy as np
import scipy.io as reader

PDE_datafile = 'cylinder_nektar_wake.mat'
PDE_datafile1 = 'cylinder_nektar_t0_vorticity.mat'
PDE_dim = 3
PDE_vars = ['x', 'y', 't']
PDE_scale = {
    'x': (1, 8),
    'y': (-2, 2),
    't': (0, 20)
}
PDE_analytic_solution = False
PDE_description = 'Ut + lambda1 (U * Ux + V * Uy) = -Px + lambda2 (Uxx + Uyy)  \n ' \
                  'Vt + lambda1 (U * Vx + V * Vy) = -Py + lambda2 (Vxx + Vyy)'
PDE_initial_condition = []
PDE_boundary_condition = ['PDE_bc1 : Ux + Vy = 0']

lambda1 = 1
lambda2 = 0.01


def PDE_bc1():
    return 'Ux + Vy = 0'


def PDE_get_data():
    data = reader.loadmat(PDE_datafile)
    #print(data)
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    return {
        'data': data,
        't': t_star,
        't_dim': len(t_star),
        'solution_type': 'X Y T U V P',
        'solution': [XX, YY, TT, UU, VV, PP]
    }


