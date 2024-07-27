import torch
import torch.nn as nn
import torch.optim as optim


class DCGAN():
    def __init__(self, nz, lr, device, channels=3):
        generator = self.Generator(nz, channels)
        discriminator = self.Discriminator(channels)
        criterion = nn.BCELoss()
        
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.criterion = criterion

        self.opt_discriminator = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)) # Same as in paper
        self.opt_generator = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999)) # Same as in paper
        
        # Apply the weights_init function to the model
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
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
        self.discriminator.zero_grad()
        batch_size = real_data.size(0)
        labels_real = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
        labels_fake = torch.full((batch_size,), 0, dtype=torch.float, device=self.device)

        output_real = self.discriminator(real_data)
        err_disc_real = self.criterion(output_real, labels_real)
        err_disc_real.backward()

        output_fake = self.discriminator(fake_data.detach())
        err_disc_fake = self.criterion(output_fake, labels_fake)
        err_disc_fake.backward()

        self.opt_discriminator.step()
        return err_disc_real + err_disc_fake

    def train_generator(self, fake_data):
        self.generator.zero_grad()
        batch_size = fake_data.size(0)
        labels = torch.full((batch_size,), 1, dtype=torch.float, device=self.device)
        output = self.discriminator(fake_data)
        err_gen = self.criterion(output, labels)
        err_gen.backward()
        self.opt_generator.step()
        return err_gen
