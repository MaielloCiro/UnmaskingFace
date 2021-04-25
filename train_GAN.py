import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import tensorflow.keras.backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing import image_dataset_from_directory as idfd
from PIL import Image
from GAN_model import *
from prepare_dataset import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

count = 0    
EPOCHS=5
LAMBDA_whole = 0.3
LAMBDA_mask = 0.7
LAMBDArc = 100

gen = generatore()
#gen.summary()
disc_whole = disc_whole_region()
#disc_whole.summary()
disc_mask = disc_mask_region()
#disc_mask.summary()
vgg_model = vgg19_model()
print("Modelli caricati")

train_ds, test_ds = prepare_tf_GAN()
print("Dati caricati")
#---------LOSS PERCEPTUAL-----------

@tf.function
def perceptual_loss(gen_image, gt_image):
    h1_list = vgg_model(gen_image)
    h2_list = vgg_model(gt_image)

    perc_loss = 0.0
    for h1, h2 in zip(h1_list, h2_list):
        h1 = K.batch_flatten(h1)
        h2 = K.batch_flatten(h2)
        perc_loss += K.sum(K.square(h1 - h2), axis=-1)
    
    return perc_loss

#---------LOSS GAN------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def disc_loss(disc_real_output, disc_gen_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output) #parte reale
    fake_loss = cross_entropy(tf.zeros_like(disc_gen_output), disc_gen_output) #parte generata
    total_loss = -(real_loss + fake_loss)
    return total_loss

def gen_loss(disc_gen_output):
    adv_loss = -cross_entropy(tf.ones_like(disc_gen_output), disc_gen_output) #loss adversarial
    return adv_loss

def rec_loss(gen_output, Igt):
    l1_loss = tf.reduce_mean(tf.abs(Igt - gen_output)) #loss L1
    SSIM_loss = 1 - tf.image.ssim(gen_output, Igt, max_val=2.0) #loss SSIM
    rc_loss = l1_loss + SSIM_loss
    return rc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_whole_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_mask_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

print("Ottimizzatori caricati")

@tf.function
def first_train_step(input_image, input_map, Igt, epoch, n):
    print("Entrato nel first a epoca "+str(epoch) + " e allo step " + str(n))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = gen([input_image, input_map], training=True)

        real_output = disc_whole(input_image, training=True)
        fake_output = disc_whole(generated_image, training=True)

        generator_loss = gen_loss(fake_output)
        discriminator_loss = disc_loss(real_output, fake_output)
        rc_loss = rec_loss(generated_image, Igt)
        perc_loss = perceptual_loss(generated_image, Igt)
        gen_tot_loss = LAMBDArc*(rc_loss + perc_loss) + generator_loss

    gradients_of_generator = gen_tape.gradient(gen_tot_loss, gen.trainable_variables)
    gradients_of_disc_whole = disc_tape.gradient(discriminator_loss, disc_whole.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_whole_optimizer.apply_gradients(zip(gradients_of_disc_whole, disc_whole.trainable_variables))

def second_train_step(input_image, input_map, Igt, epoch, n):
    print("Entrato nel second a epoca "+str(epoch) + " e allo step "+str(n))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_whole_tape, tf.GradientTape() as disc_mask_tape:
        generated_image = gen([input_image, input_map], training=True)

        real_output_whole = disc_whole(input_image, training=True)
        fake_output_whole = disc_whole(generated_image, training=True)
        real_output_mask = disc_mask([Igt, input_map, input_image], training=True)
        fake_output_mask = disc_mask([generated_image, input_map, input_image], training=True)

        gen_loss_whole = gen_loss(fake_output_whole)
        disc_loss_whole = LAMBDA_whole * disc_loss(real_output_whole, fake_output_whole)
        gen_loss_mask = gen_loss(fake_output_mask)
        disc_loss_mask = LAMBDA_mask * disc_loss(real_output_mask, fake_output_mask)
        rc_loss = rec_loss(generated_image, Igt)
        perc_loss = perceptual_loss(generated_image, Igt)
        gen_tot_loss = LAMBDArc*(rc_loss + perc_loss) + LAMBDAwhole*(gen_loss_whole) + LAMBDAmask*(gen_loss_mask)

    gradients_of_generator = gen_tape.gradient(gen_tot_loss, gen.trainable_variables)
    gradients_of_disc_whole = disc_whole_tape.gradient(disc_loss_whole, disc_whole.trainable_variables)
    gradients_of_disc_mask = disc_mask_tape.gradient(disc_loss_mask, disc_mask.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    disc_whole_optimizer.apply_gradients(zip(gradients_of_disc_whole, disc_whole.trainable_variables))
    disc_mask_optimizer.apply_gradients(zip(gradients_of_disc_mask, disc_mask.trainable_variables))

checkpoint_dir = '.\\Checkpoints\\GAN'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, disc_whole_optimizer=disc_whole_optimizer, disc_mask_optimizer=disc_mask_optimizer, generator=gen, disc_whole=disc_whole, disc_mask=disc_mask)

def fit(train_ds, epochs, test_ds):
    first_epochs = int(round(epochs * 0.4))
    for epoch in range(epochs):
        start = time.time()
        for example_input, example_map, example_target in test_ds.take(1):
            generate_images(gen, example_input, example_map, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, input_map, target) in train_ds.enumerate():
            if epoch < first_epochs:
                first_train_step(input_image, input_map, target, epoch, n)
            else:
                second_train_step(input_image, input_map, target, epoch, n)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

    checkpoint.save(file_prefix=checkpoint_prefix)

def generate_images(model, test_input, test_map, tar):
    prediction = model([test_input, test_map], training=True)
    fig=plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Test Input', 'Ground Truth', 'Predicted Image']
    global count
    count += 1
    stringa ="Risultati\\GAN\\20k_5epoch\\" + str(count) + ".png"
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
    fig.savefig(stringa)

fit(train_ds, EPOCHS, test_ds)