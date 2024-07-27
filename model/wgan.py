import torch
import torch.nn as nn
import torch.optim as optim

class WGAN():
    def __init__(self, nz, lr, device, lambda_gp=10, clamp_value=0.01, n_critic=5, channels=1):
        generator = self.Generator(nz, channels)
        discriminator = self.Discriminator(channels)
        
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.clamp_value = clamp_value
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.opt_discriminator = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        self.opt_generator = optim.RMSprop(self.generator.parameters(), lr=lr)

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
                nn.Conv2d(512, 1, 4, 1, 0, bias=False)
            )

        def forward(self, input):
            return self.main(input).view(-1)

    def clamp_weights(self):
        if self.clamp_value is not None:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clamp_value, self.clamp_value)

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device, requires_grad=True)
        interpolates = epsilon * real_data + (1 - epsilon) * fake_data
        interpolates = interpolates.to(self.device)
        interpolates.requires_grad_(True)

        d_interpolates = self.discriminator(interpolates)
        
        fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_discriminator(self, real_data, fake_data):
        self.discriminator.zero_grad()
        
        output_real = self.discriminator(real_data)
        D_real = output_real.mean()
        
        output_fake = self.discriminator(fake_data.detach())
        D_fake = output_fake.mean()
        
        gradient_penalty = self.gradient_penalty(real_data, fake_data.detach())
        
        D_loss = D_fake - D_real + self.lambda_gp * gradient_penalty
        D_loss.backward()
        
        self.opt_discriminator.step()
        
        return D_loss

    def train_generator(self, fake_data):
        self.generator.zero_grad()
        output = self.discriminator(fake_data)
        G_loss = -output.mean()
        G_loss.backward()
        self.opt_generator.step()
        return G_loss
