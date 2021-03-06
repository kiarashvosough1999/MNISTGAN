import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import torch.optim as optim
from Utils.Trainer import Trainer


class Epocher:

    def __init__(self,
                 train_loader,
                 train_data,
                 device,
                 generator,
                 discriminator,
                 trainer: Trainer,
                 nz=128,
                 epochs=200,
                 k=1,
                 sample_size=64):
        self.train_loader = train_loader
        self.train_data = train_data
        self.generator = generator
        self.trainer = trainer
        self.discriminator = discriminator
        self.device = device
        self.epochs = epochs
        self.k = k  # number of steps to apply to the discriminator
        self.nz = nz  # latent vector size
        self.sample_size = sample_size  # fixed sample size
        self.losses_g = []  # to store generator loss after each epoch
        self.losses_d = []  # to store discriminator loss after each epoch
        self.images = []  # to store images generatd by the generator

        self.optim_g = optim.Adam(generator.parameters(), lr=0.0002)  # optimizers
        self.optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)  # optimizers

    def start(self):
        noise = self.create_noise(self.sample_size, self.nz)
        for epoch in range(self.epochs):
            loss_g = 0.0
            loss_d = 0.0
            for bi, data in tqdm(enumerate(self.train_loader), total=int(len(self.train_data) / self.train_loader.batch_size)):
                image, _ = data
                image = image.to(self.device)
                b_size = len(image)
                # run the discriminator for k number of steps
                for step in range(self.k):
                    data_fake = self.generator(self.create_noise(b_size, self.nz)).detach()
                    data_real = image
                    # train the discriminator network
                    loss_d += self.trainer.train_discriminator(self.optim_d, data_real, data_fake)
                data_fake = self.generator(self.create_noise(b_size, self.nz))
                # train the generator network
                loss_g += self.trainer.train_generator(self.optim_g, data_fake)

            # create the final fake image for the epoch
            generated_img = self.generator(noise).cpu().detach()
            # make the images as grid
            generated_img = make_grid(generated_img)
            # save the generated torch tensor models to disk
            Epocher.save_generator_image(generated_img, f"outputs/gen_img{epoch}.png")
            self.images.append(generated_img)
            epoch_loss_g = loss_g / bi  # total generator loss for the epoch
            epoch_loss_d = loss_d / bi  # total discriminator loss for the epoch
            self.losses_g.append(epoch_loss_g)
            self.losses_d.append(epoch_loss_d)

            print(f"Epoch {epoch} of {self.epochs}")
            print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")

    # function to create the noise vector
    def create_noise(self, sample_size, nz):
        return torch.randn(sample_size, nz).to(self.device)

    # to save the images generated by the generator
    @staticmethod
    def save_generator_image(self, image, path):
        save_image(image, path)
