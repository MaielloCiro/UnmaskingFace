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
    score_SSIM = tf.image.ssim(prediction, tar, max_val=2.0)
    score_PSNR = tf.image.psnr(prediction, tar, max_val=2.0)
    index_SSIM = tf.argmax(score_SSIM)
    index_PSNR = tf.argmax(score_PSNR)
    np_score_PSNR=score_PSNR.numpy()
    average_PSNR = np.average(np_score_PSNR)
    np_score_SSIM=score_SSIM.numpy()
    average_SSIM = np.average(np_score_SSIM)
    fig=plt.figure(figsize=(30, 15))
    fig.text(0.5, 0.07, "SSIM: " + str(np_score_SSIM[index_SSIM]), fontsize=28, horizontalalignment="center")
    fig.text(0.5, 0.03, "PSNR: " + str(np_score_PSNR[index_PSNR]) + "dB", fontsize=28, horizontalalignment="center")
    plt.suptitle("TEST", fontsize=40, ha="center")
    display_list = [test_input[index_SSIM], tar[index_SSIM], prediction[index_SSIM], test_input[index_PSNR], tar[index_PSNR], prediction[index_PSNR]]
    title = ['SSIM_Test Input', 'SSIM_Ground Truth', 'SSIM_Predicted Image','PSNR_Test Input', 'PSNR_Ground Truth', 'PSNR_Predicted Image']
    global count
    stringa ="Risultati\\GAN\\test\\" + str(count) + ".png"
    count += 1
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.title(title[i], fontsize=22)
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    # plt.show()
    fig.savefig(stringa)



# gen = tf.keras.models.load_model("generatore.h5")
gen = generatore()
checkpoint_dir = "C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\Checkpoints\\GAN\\final_loss"
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
    generate_images(gen, example_input, example_map, example_target)

# generate_images(gen, X, Y, Z)
