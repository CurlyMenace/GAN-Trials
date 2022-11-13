import time
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, output_activation = None):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(48, 144)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(144,96)
        self.linear3 = nn.Linear(96,48)
        self.output_activation = output_activation

    def forward(self, input_tensor):
        intermediate = self.linear1(input_tensor)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.linear3(intermediate)

        if self.output_activation is not None:
            intermediate = self.output_activation(intermediate)
        return intermediate

class Discriminator(nn.Module):
    def __init__(self, input_dim, layers):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self._init_layers(layers)

    def _init_layers(self, layers):
        self.module_list = nn.ModuleList()
        last_layer = self.input_dim
        for index, width in enumerate(layers):
            self.module_list.append(nn.Linear(last_layer, width))
            last_layer = width
            if index + 1 != len(layers):
                self.module_list.append(nn.LeakyReLU())
        else:
            self.module_list.append(nn.Sigmoid())

    def forward(self, input_tensor):
        intermediate = input_tensor
        for layer in self.module_list:
            intermediate = layer(intermediate)
        return intermediate

class GAN():
    def __init__(self, generator, discriminator, noise_fn, data_fn, batch_size=200, lr=0.0002):
        self.device = torch.device("cuda:0")
        
        self.generator = generator
        self.generator = self.generator.to(self.device)
        self.discriminator = discriminator
        self.discriminator = self.discriminator.to(self.device)
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.target_ones = torch.ones((batch_size, 48)).to(self.device)
        self.target_zeros = torch.zeros((batch_size, 48)).to(self.device)
    
    def generate_samples(self, latent_vec=None, num=None):
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        return samples

    def train_step_generator(self):
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()

        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self):
        self.discriminator.zero_grad()
        # batch_x = self.batch_size * index
        # batch_y = (self.batch_size * index) + self.batch_size
        real_samples = self.data_fn(self.batch_size)
        pred_real = self.discriminator(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)

        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        loss = loss_real + loss_fake
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_step(self):
        loss_d = self.train_step_discriminator()
        loss_g = self.train_step_generator()
        return loss_g, loss_d


    def check_discriminator(self, data_real, data_fake):
        pred_real = self.discriminator(data_real)
        loss_real = self.criterion(pred_real, self.target_ones)

        pred_fake = self.discriminator(data_fake)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        print(loss_real.item(), loss_fake.item())

def main():
    from time import time 

    epochs = 10000
    generator = Generator()
    discriminator = Discriminator(48, [144, 96, 48])
    noise_fn = lambda x: torch.rand((x, 48), device='cuda:0')

    data = pd.read_csv('normalised_dataset.csv')
    #data_fn = lambda x, y: torch.tensor(data.iloc[x:y].to_numpy(), device='cuda:0').float()
    data_fn = lambda x: torch.tensor(data.sample(n=x).to_numpy(), device='cuda:0').float()

    data_real = torch.tensor(data.iloc[0:200].to_numpy(), device='cuda:0').float()
    
    gan = GAN(generator, discriminator, noise_fn, data_fn)

    data_fake = gan.generate_samples()
    loss_g, loss_d_real, loss_d_fake = [], [], []

    start = time()
    for epoch in range(epochs):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0,0,0

        # for batch in range(batches):
        lg_, (ldr_, ldf_) = gan.train_step()
        loss_g_running += lg_
        loss_d_real_running += ldr_
        loss_d_fake_running += ldf_

        loss_g.append(loss_g_running)
        loss_d_real.append(loss_d_real_running)
        loss_d_fake.append(loss_d_fake_running)
        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
            f" G={loss_g[-1]:.3f},"
            f" Dr={loss_d_real[-1]:.3f},"
            f" Df={loss_d_fake[-1]:.3f}")

        gan.check_discriminator(data_real, data_fake)

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(loss_g, label="G")
    plt.plot(loss_d_real, label="Dr")
    plt.plot(loss_d_fake, label="Df")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('fig1.png')


if __name__ == "__main__":
    main()