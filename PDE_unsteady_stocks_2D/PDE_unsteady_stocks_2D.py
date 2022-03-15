import numpy as np
import torch
from torch.autograd import Variable

PDE_datafile = ''
PDE_dim = 3
PDE_vars = ['x1', 'x2', 't']
PDE_scale = {
    'x1': (0, 1),
    'x2': (0, 1),
    't': (0, 1)
}
PDE_analytic_solution = True
PDE_description = 'V = [u1(t, x1, x2), u2(t, x1, x2)]\n' \
                  'q = p(t, x1, x2)\n' \
                  'U1_t - 0.025 * (U1_x1x1+U1_x2x2) + P_x1 = f1 \n ' \
                  'U2_t - 0.025 * (U2_x1x1+U2_x2x2) + P_x2 = f2\n' \
                  'U1_x1 + U2_x2 =0'
PDE_initial_condition = ['PDE_ic1(x1,x2, t=0): V0']
PDE_boundary_condition = ['PDE_bc1 : [U1,U2]=0 on x1=0 or x1=1 or x2=0 or x2=1']

alpha = 0.025


def f1(t, x1, x2):
    t = Variable(t, requires_grad=True)
    x1 = Variable(x1, requires_grad=True)
    x2 = Variable(x2, requires_grad=True)
    u1 = PDE_u1(t, x1, x2)
    p = PDE_p(t, x1, x2)
    u1_t = torch.autograd.grad(u1, t,grad_outputs=torch.ones_like(u1),create_graph=True)[0]
    u1_x1 = torch.autograd.grad(u1, x1,grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u1_x1x1 = torch.autograd.grad(u1_x1, x1,grad_outputs=torch.ones_like(u1_x1), create_graph=True)[0]
    u1_x2 = torch.autograd.grad(u1, x2,grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    u1_x2x2 = torch.autograd.grad(u1_x2, x2, grad_outputs=torch.ones_like(u1_x2),create_graph=True)[0]
    p_x1 = torch.autograd.grad(p, x1, grad_outputs=torch.ones_like(p),create_graph=True)[0]
    f = u1_t - 0.025 * (u1_x1x1 + u1_x2x2) + p_x1
    return f


def f2(t, x1, x2):
    t = Variable(t, requires_grad=True)
    x1 = Variable(x1, requires_grad=True)
    x2 = Variable(x2, requires_grad=True)
    u2 = PDE_u2(t, x1, x2)
    p = PDE_p(t, x1, x2)
    u2_t = torch.autograd.grad(u2, t, grad_outputs=torch.ones_like(u2),create_graph=True)[0]
    u2_x1 = torch.autograd.grad(u2, x1,grad_outputs=torch.ones_like(u2), create_graph=True)[0]
    u2_x1x1 = torch.autograd.grad(u2_x1, x1,grad_outputs=torch.ones_like(u2_x1), create_graph=True)[0]
    u2_x2 = torch.autograd.grad(u2, x2, grad_outputs=torch.ones_like(u2),create_graph=True)[0]
    u2_x2x2 = torch.autograd.grad(u2_x2, x2,grad_outputs=torch.ones_like(u2_x2), create_graph=True)[0]
    p_x2 = torch.autograd.grad(p, x2,grad_outputs=torch.ones_like(p), create_graph=True)[0][:1]
    f = u2_t - 0.025 * (u2_x1x1 + u2_x2x2) + p_x2
    return f


def PDE_ic1(x1, x2, t=0):
    return [PDE_u1(t, x1, x2), PDE_u2(t, x1, x2)]


def PDE_bc1(t, x1, x2):
    return [PDE_u1(t, x1, x2), PDE_u2(t, x1, x2)]


def PDE_u1(t, x1, x2):
    return 2 * torch.sin(t) * torch.pow(torch.sin(torch.pi * x1), 2) * torch.sin(torch.pi * x2) * torch.cos(torch.pi * x2) * torch.pi


def PDE_u2(t, x1, x2):
    return -2 * torch.sin(t) * torch.pow(torch.sin(torch.pi * x2), 2) * torch.sin(torch.pi * x1) * torch.cos(torch.pi * x1) * torch.pi


def PDE_p(t, x1, x2):
    return torch.sin(t) * torch.cos(torch.pi * x1) * torch.cos(torch.pi * x2)
