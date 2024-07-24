import torch
import torch.nn as nn
import torch.optim as optim
import os

class GANBase:
    def __init__(self, generator, discriminator, nz, lr, device, criterion=None, clamp_value=None, n_critic=1):
        self.device = device
        self.netG = generator.to(device)
        self.netD = discriminator.to(device)
        self.criterion = criterion
        self.clamp_value = clamp_value
        self.n_critic = n_critic

        if criterion is None:  # WGAN
            self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=lr)
            self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=lr)
        else:  # DCGAN
            self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
            self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))

    def train_discriminator(self, real_data, fake_data):
        raise NotImplementedError

    def train_generator(self, fake_data):
        raise NotImplementedError

    def clamp_weights(self):
        if self.clamp_value is not None:
            for p in self.netD.parameters():
                p.data.clamp_(-self.clamp_value, self.clamp_value)

    def save_model(self, path, gan_type):
        torch.save(self.netG.state_dict(), os.path.join(path, f"generator_{gan_type}.pth"))
        torch.save(self.netD.state_dict(), os.path.join(path, f"discriminator_{gan_type}.pth"))

    def load_model(self, path, gan_type):
        self.netG.load_state_dict(torch.load(os.path.join(path, f"generator_{gan_type}.pth")))
        self.netD.load_state_dict(torch.load(os.path.join(path, f"discriminator_{gan_type}.pth")))
