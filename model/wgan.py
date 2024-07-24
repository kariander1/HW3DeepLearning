import torch
import torch.nn as nn
import torch.optim as optim

class GANBase(nn.Module):
    def __init__(self, generator, discriminator, nz, lr, device, clamp_value=0.01, n_critic=5):
        super().__init__()
        self.netG = generator.to(device)
        self.netD = discriminator.to(device)
        self.nz = nz
        self.lr = lr
        self.device = device
        self.clamp_value = clamp_value
        self.n_critic = n_critic
        self.optimizerG = optim.RMSprop(self.netG.parameters(), lr=lr)
        self.optimizerD = optim.RMSprop(self.netD.parameters(), lr=lr)

    def clamp_weights(self):
        for p in self.netD.parameters():
            p.data.clamp_(-self.clamp_value, self.clamp_value)

class WGAN(GANBase):
    def __init__(self, nz, lr, device, clamp_value=0.01, n_critic=5, channels=1):
        generator = self.Generator(nz, channels)
        discriminator = self.Discriminator(channels)
        super().__init__(generator, discriminator, nz, lr, device, clamp_value=clamp_value, n_critic=n_critic)

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
                nn.Conv2d(512, 1, 4, 1, 0, bias=False)
            )

        def forward(self, input):
            return self.main(input).view(-1)

    def train_discriminator(self, real_data, fake_data):
        self.netD.zero_grad()
        output_real = self.netD(real_data)
        D_real = output_real.mean()
        D_real.backward()

        output_fake = self.netD(fake_data.detach())
        D_fake = output_fake.mean()
        D_fake.backward()

        self.optimizerD.step()
        self.clamp_weights()
        return D_real - D_fake

    def train_generator(self, fake_data):
        self.netG.zero_grad()
        output = self.netD(fake_data)
        G_loss = -output.mean()
        G_loss.backward()
        self.optimizerG.step()
        return G_loss

if __name__ == "__main__":
    # Hyperparameters
    nz = 100
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clamp_value = 0.01
    n_critic = 5
    channels = 3
    batch_size = 64
    image_size = 28

    # Initialize WGAN model
    wgan = WGAN(nz, lr, device, clamp_value, n_critic, channels)

    # Create a batch of real data (e.g., random noise for simplicity here)
    real_data = torch.randn(batch_size, channels, image_size, image_size, device=device)

    # Create a batch of fake data
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake_data = wgan.netG(noise)

    # Training step
    for _ in range(n_critic):
        d_loss = wgan.train_discriminator(real_data, fake_data)
    g_loss = wgan.train_generator(fake_data)

    print(f"D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")
