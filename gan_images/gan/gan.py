import os
from tqdm import tqdm

import tensorflow as tf

from gan_images.models.discriminator import init_discriminator, d_loss
from gan_images.models.generator import init_generator, g_loss
from gan_images.data.data import load_real_samples, data_preprocessing


class GAN():
    def __init__(self):
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

    def train(self, data):
        EPOCHS = 50
        self.noise_dim = 100
        self.num_examples_to_generate = 16

        for epoch in range(EPOCHS):
            for batch in tqdm(data):
                self.batch_step(batch)


    @tf.function
    def batch_step(self, batch_images):

        # The @tf.function decorator forces the function to be compiled

        noise = tf.random.normal([batch_images.shape[0], self.noise_dim])

        with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            fake_samples = self.G(noise, training = True)

            fake_pred = self.D(fake_samples, training = True)
            real_pred = self.D(batch_images, training = True)

            G_loss = g_loss(fake_pred)
            D_loss = d_loss(real_pred, fake_pred)

        G_gradients = G_tape.gradient(G_loss, self.G.trainable_variables)
        D_gradients = D_tape.gradient(D_loss, self.D.trainable_variables)

        self.G_optim.apply_gradients(zip(G_gradients, self.G.trainable_variables))
        self.D_optim.apply_gradients(zip(D_gradients, self.D.trainable_variables))