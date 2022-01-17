import matplotlib
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from Discriminator.Discriminator import Discriminator
from Epocher.Epocher import Epocher
from Generator.Generator import Generator
from Utils.MNISTLoader import MNISTLoader
from Utils.Plotter import Plotter
from Utils.Trainer import Trainer

to_pil_image = transforms.ToPILImage()
matplotlib.style.use('ggplot')

nz = 128  # latent vector size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, train_data = MNISTLoader().load_MNIST()

generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)

print('***** GENERATOR *****')
print(generator)
print('*************************')

print('\n***** DISCRIMINATOR *****')
print(discriminator)
print('*************************')


# loss function
criterion = nn.BCELoss()

trainer = Trainer(generator=generator,
                  discriminator=discriminator,
                  device=device,
                  criterion=criterion)

epocher = Epocher(train_loader=train_loader,
                  train_data=train_data,
                  device=device,
                  generator=generator,
                  discriminator=discriminator,
                  trainer=trainer)

generator.train()
discriminator.train()

epocher.start()

print('TRAINING IS Finished')
torch.save(generator.state_dict(), 'outputs/generator.pth')

Plotter(epocher.losses_g, epocher.losses_d, epocher.images).plot()
