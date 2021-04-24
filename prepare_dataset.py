import tensorflow as tf
from tensorflow.keras.preprocessing import *

images = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\mascherine"
images_gt = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\mappette"
# dsize = (256, 256)
dsize = (128, 128)

def normalize(immagine):
    immagine = immagine / 255.0
    return immagine

def prepare_tf():
    X_train = image_dataset_from_directory(images, image_size=(128, 128), color_mode="grayscale", label_mode=None, validation_split=0.2, subset="training", shuffle=True, seed = 1, interpolation="lanczos5")
    X_val = image_dataset_from_directory(images, image_size=(128, 128), color_mode="grayscale", label_mode=None, validation_split=0.2, subset="validation", shuffle=True, seed = 1, interpolation="lanczos5")
    Y_train = image_dataset_from_directory(images_gt, image_size=(128, 128), color_mode="grayscale", label_mode=None, validation_split=0.2, subset="training",shuffle=True, seed = 1, interpolation="lanczos5")
    Y_val = image_dataset_from_directory(images_gt, image_size=(128, 128), color_mode="grayscale", label_mode=None, validation_split=0.2, subset="validation", shuffle=True, seed = 1, interpolation="lanczos5")
    X_train = X_train.map(normalize)
    Y_train = Y_train.map(normalize)
    dataset = tf.data.Dataset.zip((X_train , Y_train))
    X_val = X_train.map(normalize)
    Y_val = Y_train.map(normalize)
    dataval = tf.data.Dataset.zip((X_val , Y_val))
    return dataset, dataval