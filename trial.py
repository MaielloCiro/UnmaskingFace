import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from GAN_model import *
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing import *


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

def binarization(n, map):
    for i in range(0, 128):
        for j in range(0, 128):
            if map[i, j]<n:
                map[i, j] = 0
            else:
                map[i, j] = 1

def filter(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def normalize_GAN(immagine):
    immagine = (immagine / 127.5) - 1
    return immagine

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

seg_model = load_model("segmentation_model_20k_20epoch.h5")

#----SEGMENTATION-----

def map_generation(image):
    prediction = seg_model.predict(x = image)
    example = prediction[0, :, :, 0]
    binarization(0.01, example)
    example = filter(example)
    example = np.asarray(example)
    example = example.reshape(128, 128, 1)
    # stringa = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST\\145997_surgical_segmentation.jpg"
    # fig = plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(numpyimage)
    # plt.title("Original")
    # plt.subplot(1, 2, 2)
    # plt.imshow(example, cmap='gray')
    # plt.title("Segmentation")
    # fig.savefig(stringa)
    tf.keras.preprocessing.image.save_img('C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_MAP\\test\\ue.png', example)
    # cv2.imwrite('C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_MAP\\test\\ue.png', example)
    return example

def generate_image(model, test_input, test_map):
    prediction = model([test_input, test_map], training=True)
    fig=plt.figure(figsize=(15, 7.5))
    plt.suptitle("TEST", fontsize=24, ha="center")
    display_list = [test_input[0, :, :, :], test_map[0, :, :, :], prediction[0, :, :, :]]
    title = ['Test Input', 'Segmentation Map', 'Predicted Image']
    stringa ="C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\OUTPUT.png"
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i], fontsize=18)
        # getting the pixel values between [0, 1] to plot it.
        if i == 1:
            plt.imshow(display_list[i], cmap = 'gray')
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    fig.show()
    fig.savefig(stringa)

dsize = (128, 128)
path = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_MASK\\test\\039692_N95.jpg"
image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=dsize, interpolation="lanczos")
numpyimage = np.array(image)
original_image = numpyimage / 255.0
original_image = original_image.reshape(1, 128, 128, 1)
map = map_generation(original_image)
pathmask = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_MASK"
imagesmask = image_dataset_from_directory(pathmask, image_size=(128, 128), label_mode=None, shuffle=True, seed=57, interpolation="lanczos5")
imagesmask = imagesmask.map(normalize_GAN)
pathmap = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_MAP"
map = image_dataset_from_directory(pathmap, image_size=(128, 128), color_mode='grayscale', label_mode=None, shuffle=True, seed=57, interpolation="lanczos5")
map = map.map(normalize_GAN)
testset = tf.data.Dataset.zip((imagesmask,map))
for n, (example_input, example_map) in testset.enumerate():
    generate_image(gen, example_input, example_map)
# generate_image(gen, original_image, map)
