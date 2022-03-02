import os
from tqdm import tqdm
import time
import sys
import logging

import matplotlib.pyplot as plt
import tensorflow as tf

from gan_images.utils import make_dir_if_not_exists
from gan_images.models.discriminator import init_discriminator, d_loss
from gan_images.models.generator import init_generator, g_loss

logger = logging.getLogger(__name__)

class GAN():
    def __init__(
        self,
        I = (128, 128, 3)
    ):
        logger.info('instantiated GAN object')
        G_lr = 1e-4
        D_lr = 1e-4

        self.G = init_generator(image_shape = (I[0], I[1]), num_channels = I[2])
        self.D = init_discriminator(image_shape = I[0], I[1], num_channels = I[2])

        self.G_optim = tf.keras.optimizers.Adam(G_lr)
        self.D_optim = tf.keras.optimizers.Adam(D_lr)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer = self.G_optim,
            discriminator_optimizer = self.D_optim,
            generator = self.G,
            discriminator = self.G
        )
        self.checkpoint_dir = './training_checkpoints'
        make_dir_if_not_exists(self.checkpoint_dir)
        self.examples_images_dir = './training_images'
        make_dir_if_not_exists(self.examples_images_dir)

    def save_checkpoints(self):
        checkpoint_prefix= os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint.save(checkpoint_prefix)

    def train(
        self, 
        data : tf.data.Dataset, 
        num_epochs : int = 50, 
        mode : str = 'simultaneous',
        batch_size : int = 64,
        min_acc : float = 0.8,
        noise_dim : int = 100,
        num_examples_to_generate : int = 16
    ):
        self.noise_dim = noise_dim

        for epoch in range(num_epochs):
            # ---------------------
            start = time.time()
            logger.info(f'epoch {epoch}')
            
            logger.info(f'training in {mode} mode')
            if mode == 'alternate':
                self.train_D(data, batch_size = batch_size, max_epochs = 5, min_acc = min_acc)
                self.train_G(batch_size = batch_size)
            elif mode == 'simultaneous':
                self.simultanous_training(data)
            else:
                logger.error("wrong  mode argument, can either be 'simultaneous' or 'alternate'")
                sys.exit()

            self.save_checkpoints()
            inputs = tf.random.normal([num_examples_to_generate, self.noise_dim])
            self.generate_and_save_images(inputs)
            logger.info(f"End of epoch {epoch}: {time.time() - start} seconds")
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


    def generate_and_save_images(self, epoch, test_input):
        predictions = self.G(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0])
            plt.axis('off')

        plt.savefig(
            os.path.join(
                self.examples_images_dir,
                'images_at_epoch_{:04d}.png'.format(epoch))
        )