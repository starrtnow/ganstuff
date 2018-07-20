import torch
import os
from torch.autograd import Variable, grad

def condense_range(tensor):
    return (tensor + 1) * 0.5

def expand_range(tensor):
    return (tensor * 2) - 1

def gradient_penalty(x, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = (torch.rand(shape)).cuda()
    beta = (torch.rand(x.size())).cuda()

    y = x + 0.5 * x.std() * beta
    z = x + alpha * (y - x)

    # gradient penalty
    z = (Variable(z, requires_grad=True)).cuda()
    o = f(z)
    g = grad(o, z, grad_outputs=(torch.ones(o.size())).cuda(), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    
    return gp

def save(network, name, path = "saved_networks"):
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, name)
    torch.save(network.state_dict(), path)

def load(network, name, path = "saved_networks"):
    path = os.path.join(path, name)
    network.load_state_dict(torch.load(path))
   

