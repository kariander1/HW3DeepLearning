import torch
import torch.nn as nn
from .gan_base import GANBase

class DCGAN(GANBase):
    def __init__(self, nz, lr, device, channels=3):
        generator = self.Generator(nz, channels)
        discriminator = self.Discriminator(channels)
        criterion = nn.BCELoss()
        super().__init__(generator, discriminator, nz, lr, device, criterion=criterion)

    class Generator(nn.Module):
        def __init__(self, nz, channels):
            super().__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            return self.main(input)

    class Discriminator(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input).view(-1)

    def train_discriminator(self, real_data, fake_data):
        self.netD.zero_grad()
        batch_size = real_data.size(0)
        labels_real = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
        labels_fake = torch.full((batch_size,), 0, dtype=torch.float, device=self.device)

        output_real = self.netD(real_data)
        errD_real = self.criterion(output_real, labels_real)
        errD_real.backward()

        output_fake = self.netD(fake_data.detach())
        errD_fake = self.criterion(output_fake, labels_fake)
        errD_fake.backward()

        self.optimizerD.step()
        return errD_real + errD_fake

    def train_generator(self, fake_data):
        self.netG.zero_grad()
        batch_size = fake_data.size(0)
        labels = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
        output = self.netD(fake_data)
        errG = self.criterion(output, labels)
        errG.backward()
        self.optimizerG.step()
        return errG
