from torchvision.transforms import transforms
import numpy as np
from matplotlib import pyplot as plt
import imageio


class Plotter:

    def __init__(self, generator_losses, discrim_losses, images):
        self._generator_losses = generator_losses
        self._discrim_losses = discrim_losses
        self._images = images

    def plot(self):
        to_pil_image = transforms.ToPILImage()
        # save the generated images as GIF file
        imgs = [np.array(to_pil_image(img)) for img in self._images]
        imageio.mimsave('outputs/generator_images.gif', imgs)

        # plot and save the generator and discriminator loss
        plt.figure()
        plt.plot(self._generator_losses, label='Generator loss')
        plt.plot(self._discrim_losses, label='Discriminator Loss')
        plt.legend()
        plt.savefig('outputs/loss.png')
