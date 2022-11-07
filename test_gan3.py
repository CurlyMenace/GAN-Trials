import torch
from torch import nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, output_activation=None):
        """A generator for mapping a latent space to a sample space.
            Args:
                latent_dim (int): latent dimension ("noise vector")
                layers (List[int]): A list of layer widths including output width
                output_activation: torch activation function or None
        """

        super(Generator, self).__init__()
        self.linear1 = nn.Linear(latent_dim, 64)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,1)
        self.output_activation = output_activation

    def forward(self, input_tensor):
        """Forward pass; map latent vectors to samples."""
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
        """A discriminator for discerning real from generated samples.
        
        params: 
            input_dim (int): width of the input
            layers (List[int]): a list of layer widths including output width
        
        Output activation is Sigmoid.
        """

        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self._init_layers(layers)

    def _init_layers(self, layers):
        """Initialize the layers and store as self.module_list."""

        self.module_list = nn.ModuleList()
        last_layer = self.input_dim
        for index, width in enumerate(layers):
            print(index,width)
            self.module_list.append(nn.Linear(last_layer, width))
            print("In for loop")
            last_layer = width
            if index + 1 != len(layers):
                print("Appending leakyrelu")
                self.module_list.append(nn.LeakyReLU())
        else:
            print("sigmoid")
            self.module_list.append(nn.Sigmoid())

    def forward(self, input_tensor):
        """Forward pass; map samples to confidence they are real [0, 1]."""
        intermediate = input_tensor
        for layer in self.module_list:
            intermediate = layer(intermediate)
        return intermediate
        

class VanillaGAN():
    def __init__(self, generator, discriminator, noise_fn, data_fn,
                batch_size=32, lr_d=1e-3, lr_g=2e-4):
        """A GAN class for holding and training a generator and discriminator
                Args:
                    generator: a Ganerator network
                    discriminator: A Discriminator network
                    noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
                    data_fn: function f(num: int) -> pytorch tensor, (real samples)
                    batch_size: training batch size
                    device: cpu or CUDA
                    lr_d: learning rate for the discriminator
                    lr_g: learning rate for the generator
        """
        self.device = torch.device("cuda:0")

        self.generator = generator
        self.generator = self.generator.to(self.device)
        self.discriminator = discriminator
        self.discriminator = self.discriminator.to(self.device)
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size
        self.criterion = nn.BCELoss()
        self.optim_d = optim.Adam(discriminator.parameters(), 
                                    lr=lr_d, betas=(0.5, 0.999))
        self.optim_g = optim.Adam(generator.parameters(),
                                    lr=lr_g, betas=(0.5, 0.999))

        self.target_ones = torch.ones((batch_size, 1)).to(self.device)
        self.target_zeros = torch.zeros((batch_size, 1)).to(self.device)
        print(self.target_ones)
    def generate_samples(self, latent_vec=None, num=None):
        """Sample from the generator.
        
        Args: 
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
            
        If latent_vec and num are None then use self.batch_size random latent vectors."""

        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        with torch.no_grad():
            samples = self.generator(latent_vec)
        return samples

    def train_step_generator(self):
        """Train the generator one step and return the loss."""
        self.generator.zero_grad()

        latent_vec = self.noise_fn(self.batch_size)
        generated = self.generator(latent_vec)
        classifications = self.discriminator(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()

        self.optim_g.step()
        return loss.item()

    def train_step_discriminator(self):
        """Train the discriminator one step and return the losses."""

        self.discriminator.zero_grad()

        #real samples
        real_samples = self.data_fn(self.batch_size)
        pred_real = self.discriminator(real_samples)
        print(pred_real.size(), self.target_ones.size())
        loss_real = self.criterion(pred_real, self.target_ones)

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        with torch.no_grad():
            fake_samples = self.generator(latent_vec)
        pred_fake = self.discriminator(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)        

        # combine
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_d.step()
        return loss_real.item(), loss_fake.item()

    def train_step(self):
        """Train both networks and return the losses."""
        loss_d = self.train_step_discriminator()
        loss_g = self.train_step_generator()
        return loss_g, loss_d


def main():
    from time import time
    epochs = 600
    batches = 10
    generator = Generator(1)
    discriminator = Discriminator(1, [64, 32, 1])
    noise_fn = lambda x: torch.rand((x, 1), device='cuda:0')
    data_fn = lambda x: torch.randn((x, 1), device='cuda:0')
    gan = VanillaGAN(generator, discriminator, noise_fn, data_fn)
    loss_g, loss_d_real, loss_d_fake = [], [], []
    start = time()
    for epoch in range(epochs):
        loss_g_running, loss_d_real_running, loss_d_fake_running = 0,0,0

        for batch in range(batches):
            lg_, (ldr_, ldf_) = gan.train_step()
            loss_g_running += lg_
            loss_d_real_running += ldr_
            loss_d_fake_running += ldf_

        loss_g.append(loss_g_running / batches)
        loss_d_real.append(loss_d_real_running / batches)
        loss_d_fake.append(loss_d_fake_running / batches)
        print(f"Epoch {epoch+1}/{epochs} ({int(time() - start)}s):"
              f" G={loss_g[-1]:.3f},"
              f" Dr={loss_d_real[-1]:.3f},"
              f" Df={loss_d_fake[-1]:.3f}")

if __name__ == "__main__":
    main()