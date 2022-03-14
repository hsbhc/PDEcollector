import numpy as np
import torch

PDE_datafile = ''
PDE_dim = 4
PDE_vars = ['x1', 'x2', 'x3', 't']
PDE_scale = {
    'x1': (0, 1),
    'x2': (0, 1),
    'x3': (0, 1),
    't': (0, 1)
}
PDE_analytic_solution = True
PDE_description = 'see pdf'
PDE_initial_condition = ['PDE_ic1(x1,x2,x3 t=0): V0']
PDE_boundary_condition = ['PDE_bc1 : [U1,U2,U3]=0']

alpha = 0.025


def f1(t, x1, x2, x3):
    u1 = PDE_u1(t, x1, x2, x3)
    p = PDE_p(t, x1, x2, x3)
    u1_t = torch.autograd.grad(u1, t, create_graph=True)[0]
    u1_x1 = torch.autograd.grad(u1, x1, create_graph=True)[0]
    u1_x1x1 = torch.autograd.grad(u1_x1, x1, create_graph=True)[0]
    u1_x2 = torch.autograd.grad(u1, x2, create_graph=True)[0]
    u1_x2x2 = torch.autograd.grad(u1_x2, x2, create_graph=True)[0]
    u1_x3 = torch.autograd.grad(u1, x3, create_graph=True)[0]
    u1_x3x3 = torch.autograd.grad(u1_x3, x3, create_graph=True)[0]
    p_x1 = torch.autograd.grad(p, x1, create_graph=True)[0]
    f = u1_t - 0.025 * (u1_x1x1 + u1_x2x2 + u1_x3x3) + p_x1
    return f


def f2(t, x1, x2, x3):
    u2 = PDE_u2(t, x1, x2, x3)
    p = PDE_p(t, x1, x2, x3)
    u2_t = torch.autograd.grad(u2, t, create_graph=True)[0]
    u2_x1 = torch.autograd.grad(u2, x1, create_graph=True)[0]
    u2_x1x1 = torch.autograd.grad(u2_x1, x1, create_graph=True)[0]
    u2_x2 = torch.autograd.grad(u2, x2, create_graph=True)[0]
    u2_x2x2 = torch.autograd.grad(u2_x2, x2, create_graph=True)[0]
    u2_x3 = torch.autograd.grad(u2, x3, create_graph=True)[0]
    u2_x3x3 = torch.autograd.grad(u2_x3, x3, create_graph=True)[0]
    p_x2 = torch.autograd.grad(p, x2, create_graph=True)[0][:1]
    f = u2_t - 0.025 * (u2_x1x1 + u2_x2x2 + u2_x3x3) + p_x2
    return f


def f3(t, x1, x2, x3):
    u3 = PDE_u3(t, x1, x2, x3)
    p = PDE_p(t, x1, x2, x3)
    u3_t = torch.autograd.grad(u3, t, create_graph=True)[0]
    u3_x1 = torch.autograd.grad(u3, x1, create_graph=True)[0]
    u3_x1x1 = torch.autograd.grad(u3_x1, x1, create_graph=True)[0]
    u3_x2 = torch.autograd.grad(u3, x2, create_graph=True)[0]
    u3_x2x2 = torch.autograd.grad(u3_x2, x2, create_graph=True)[0]
    u3_x3 = torch.autograd.grad(u3, x3, create_graph=True)[0]
    u3_x3x3 = torch.autograd.grad(u3_x3, x3, create_graph=True)[0]
    p_x3 = torch.autograd.grad(p, x3, create_graph=True)[0][:1]
    f = u3_t - 0.025 * (u3_x1x1 + u3_x2x2 + u3_x2x2 + u3_x3x3) + p_x3
    return f


def PDE_ic1(x1, x2, x3, t=0):
    return [PDE_u1(t, x1, x2, x3), PDE_u2(t, x1, x2, x3)]


def PDE_bc1(t, x1, x2, x3):
    return [PDE_u1(t, x1, x2, x3), PDE_u2(t, x1, x2, x3)]


def PDE_u1(t, x1, x2, x3):
    return torch.sin(t) * torch.sin(torch.pi * x1) ** 2 * (
            torch.sin(2 * torch.pi * x2) * torch.sin(torch.pi * x3) ** 2) - torch.sin(
        torch.pi * x2) ** 2 * torch.sin(2 * torch.pi * x3)


def PDE_u2(t, x1, x2, x3):
    return torch.sin(t) * torch.sin(torch.pi * x2) ** 2 * (
            torch.sin(2 * torch.pi * x3) * torch.sin(torch.pi * x1) ** 2) - torch.sin(
        torch.pi * x3) ** 2 * torch.sin(2 * torch.pi * x1)


def PDE_u3(t, x1, x2, x3):
    return torch.sin(t) * torch.sin(torch.pi * x3) ** 2 * (
            torch.sin(2 * torch.pi * x1) * torch.sin(torch.pi * x2) ** 2) - torch.sin(
        torch.pi * x1) ** 2 * torch.sin(2 * torch.pi * x2)


def PDE_p(t, x1, x2, x3):
    return torch.sin(t) * torch.sin(torch.pi * x1) * torch.sin(torch.pi * x2) * torch.cos(torch.pi * x3)
