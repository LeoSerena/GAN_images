import os
from tqdm import tqdm
import time
import sys
import logging

import tensorflow as tf

from gan_images.models.discriminator import init_discriminator, d_loss
from gan_images.models.generator import init_generator, g_loss

logger = logging.getLogger(__name__)

class GAN():
    def __init__(self):
        logger.info('instantiated GAN object')
        I = (128, 128)
        G_lr = 1e-4
        D_lr = 1e-4

        self.G = init_generator(image_shape = I)
        self.D = init_discriminator(image_shape = I)

        self.G_optim = tf.keras.optimizers.Adam(G_lr)
        self.D_optim = tf.keras.optimizers.Adam(D_lr)

    def save_checkpoints(self):
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            generator_optimizer = self.generator_optimizer,
            discriminator_optimizer = self.discriminator_optimizer,
            generator = self.generator,
            discriminator = self.discriminator
        )

    def train(
        self, 
        data : tf.data.Dataset, 
        num_epochs : int = 50, 
        mode : str = 'simultaneous'
    ):
        self.noise_dim = 100

        for epoch in range(num_epochs):
            # ---------------------
            start = time.time()
            print(f'epoch {epoch}')
            
            logger.info(f'training in {mode} mode')
            if mode == 'alternate':
                self.train_D(data, batch_size = 64, max_epochs = 5, min_acc = 0.8)
                self.train_G(batch_size = 64)
            elif mode == 'simultaneous':
                self.simultanous_training(data)
            else:
                sys.exit('wrong mode argument')

            print(f"End of epoch {epoch}: {time.time() - start} seconds")
            # ---------------------

    def train_D(self, data, batch_size, max_epochs, min_acc):
        logger.info('training discriminator')
        for epoch in range(max_epochs):
            num_correct = 0
            total = 0
            for batch in tqdm(data):
                noise = tf.random.normal([batch_size, self.noise_dim])

                with tf.GradientTape() as D_tape:
                    fake_samples = self.G(noise, training = True)

                    fake_pred = self.D(fake_samples, training = True)
                    real_pred = self.D(batch, training = True)

                    D_loss = d_loss(real_pred, fake_pred)

                num_correct += sum((tf.math.sigmoid(fake_pred) <= 1/2).numpy()) + sum((tf.math.sigmoid(real_pred) > 1/2).numpy())
                total += batch.shape[0]

                D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

                self.D_optim.apply_gradients(zip(D_gradients, self.D.trainable_variables))

            accuracy = num_correct / total
            logger.info(f'discriminator accuracy at {epoch} : {accuracy}')
            if accuracy > min_acc:
                break

        logger.info('finished training discriminator')

    def train_G(self, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])

        with tf.GradientTape() as G_tape:
            fake_samples = self.G(noise, training = True)

            fake_pred = self.D(fake_samples, training = True)

            G_loss = g_loss(fake_pred)

        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)

        self.G_optim.apply_gradients(zip(G_gradients, self.G.trainable_variables))

    @tf.function
    def simultanous_training(self, data, batch_size = 64):
        # The @tf.function decorator forces the function to be compiled
        # This function is weird as it trains the two models at the same time while the
        # paper suggest to make the discriminator converge before training the generator every time...
        for batch in tqdm(data):
            noise = tf.random.normal([batch_size, self.noise_dim])

            with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
                fake_samples = self.G(noise, training = True)

                fake_pred = self.D(fake_samples, training = True)
                real_pred = self.D(batch, training = True)

                G_loss = g_loss(fake_pred)
                D_loss = d_loss(real_pred, fake_pred)

            G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
            D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

            self.G_optim.apply_gradients(zip(G_gradients, self.G.trainable_variables))
            self.D_optim.apply_gradients(zip(D_gradients, self.D.trainable_variables))