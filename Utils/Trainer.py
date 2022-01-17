import torch


class Trainer:

    def __init__(self, generator, discriminator, device, criterion):
        self._generator = generator
        self._discriminator = discriminator
        self._device = device
        self._criterion = criterion

    def _label_real(self, size):
        data = torch.ones(size, 1)
        return data.to(self._device)

    def _label_fake(self, size):
        data = torch.zeros(size, 1)
        return data.to(self._device)

    def train_generator(self, optimizer, data_fake):
        b_size = data_fake.size(0)
        real_label = self._label_real(b_size)

        optimizer.zero_grad()

        output = self._discriminator(data_fake)
        loss = self._criterion(output, real_label)

        loss.backward()
        optimizer.step()

        return loss

    def train_discriminator(self, optimizer, data_real, data_fake):
        b_size = data_real.size(0)
        real_label = self._label_real(b_size)
        fake_label = self._label_fake(b_size)

        optimizer.zero_grad()

        output_real = self._discriminator(data_real)
        loss_real = self._criterion(output_real, real_label)

        output_fake = self._discriminator(data_fake)
        loss_fake = self._criterion(output_fake, fake_label)

        loss_real.backward()
        loss_fake.backward()
        optimizer.step()

        return loss_real + loss_fake

