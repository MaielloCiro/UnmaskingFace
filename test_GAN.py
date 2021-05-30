import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import datetime
import tensorflow.keras.backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing import image_dataset_from_directory as idfd
from PIL import Image
from GAN_model import *
from prepare_dataset import *
counter = 0
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

def normalize(immagine):
    immagine = immagine / 127.5 - 1
    return immagine

def denormalize(immagine):
    immagine = (immagine * 1.725) - 1
    return immagine

def generate_images(model, test_input, test_map, tar):
    prediction = model([test_input, test_map], training=False)
    prediction = denormalize(prediction)
    fig=plt.figure(figsize=(15, 15))
    plt.suptitle("Test")
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Test Input', 'Ground Truth', 'Predicted Image']
    global counter
    counter += 1
    stringa ="C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\Risultati\\GAN\\test100epoch_seed57\\" + str(counter) + ".png"
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.show()
    fig.savefig(stringa)


# gen = tf.keras.models.load_model("generatore.h5")
gen = generatore()
checkpoint_dir = "C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\Checkpoints\\GAN"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=gen)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
checkpoint.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

dsize = (128, 128)
pathmask = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\testmasked1k"
pathmap = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\testmaps1k"
pathgt = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\testgt1k"
imagesmask = image_dataset_from_directory(pathmask, image_size=(128, 128), label_mode=None, shuffle=True, seed=57, interpolation="lanczos5")
imagesmap = image_dataset_from_directory(pathmap, image_size=(128, 128), color_mode="grayscale", label_mode=None, shuffle=True, seed=57, interpolation="lanczos5")
imagesgt = image_dataset_from_directory(pathgt, image_size=(128, 128), label_mode=None, shuffle=True, seed=57, interpolation="lanczos5")
imagesmask = imagesmask.map(normalize)
imagesmap = imagesmap.map(normalize)
imagesgt = imagesgt.map(normalize)
testset = tf.data.Dataset.zip((imagesmask, imagesmap, imagesgt))
# numpyimagemask = np.array(imagemask)
# X = numpyimagemask / 127.5 - 1
# X = X.reshape(1, 128, 128, 3)
# numpyimagemap = np.array(imagemap)
# Y = numpyimagemap / 127.5 - 1
# Y = Y.reshape(1, 128, 128, 1)
# numpyimagegt = np.array(imagegt)
# Z = numpyimagegt / 127.5 - 1
# Z = Z.reshape(1, 128, 128, 3)
count=1
for n, (example_input, example_map, example_target) in testset.enumerate():
    print(count)
    count += 1
    generate_images(gen, example_input, example_map, example_target)

# generate_images(gen, X, Y, Z)
