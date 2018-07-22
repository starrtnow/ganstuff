from DCGAN.networks import Gen, Disc, weights_init
from util import condense_range, expand_range, save, load
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torch.functional as f
import torchvision
import time

#parameters
batch_size = 32
n_epochs = 10000

#Create Generator and Discriminator
G = Gen()
D = Disc(batch_size)

weights_init(G)
weights_init(D)

G = G.cuda()
D = D.cuda()

opt_G = torch.optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr = 0.0002, betas = (0.5, 0.999))
criterion = nn.BCELoss()

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

#data
data = torchvision.datasets.ImageFolder('./danbooru/data/d_chibi', transform = torchvision.transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 4)

def sample(uid = ""):
    noise = torch.randn(10, 100, 1, 1).cuda()
    fake = condense_range(G(noise))
    torchvision.utils.save_image(fake, "./saved_images/{}.png".format(uid))

REAL = 1
FAKE = 0
CHECKPOINT = 50

for n in range(0, n_epochs):
    print("Epoch: {}".format(n))
    for img, _ in loader:
        #Update D
        D.zero_grad()
        X = Variable(img).cuda()
        
        label = torch.full((batch_size, 1), REAL).cuda()
        output = D(X)
        err_real = criterion(output, label)
        err_real.backward()
        D_x = output.mean().item()
        
        #Train with fake
        noise = torch.randn(batch_size, 100, 1, 1).cuda()
        fake = condense_range(G(noise))
        label = torch.full((batch_size, 1), FAKE).cuda()
        output = D(fake.detach())
        err_fake = criterion(output, label)
        err_fake.backward()
        D_g_x = output.mean().item()
        
        gp = gradient_penalty(X.data, D)
        gp.backward()
        
        err_D = err_real + err_fake
        opt_D.step()
        
        
        
        
        #Update generator
        G.zero_grad()
        label = torch.full((batch_size, 1), REAL).cuda()
        output = D(fake)
        err_G = criterion(output, label)
        err_G.backward()
        D_g_z = output.mean().item()
        opt_G.step()
    if n % CHECKPOINT == 0:
        #first print loss
        print("Discriminator loss: ", err_D + gp)
        print("Generator Loss: ", err_G)
        
        #save model
        uid = time.time()
        save(G, "{}.gen".format(uid))
        save(D, "{}.disc".format(uid))
        
        #sample image
        sample(uid)
        