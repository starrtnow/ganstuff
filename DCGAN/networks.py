import torch
import torch.nn as nn
import torch.functional as f


NZ = 100 #latent vector size
NGF = 64 #channel scale factor for generator
NDF = 64 #channel scale factor for discriminator
NC = 3 #channel of the images

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Gen(nn.Module):

    def __init__(self):
        super(Gen, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ , NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 2, NGF * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 1),
            nn.ReLU(True),

            nn.ConvTranspose2d(NGF * 1, NC, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, X):
        output = self.main(X)
        return output

class Disc(nn.Module):
    
    def __init__(self):
        super(Disc, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 8, NDF * 16, 4, 2, 1, bias=True),
            nn.BatchNorm2d(NDF * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(NDF * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X):
        output = self.main(X)
        return output.view(-1, 1)
