'''
Preparation dataset for both modules
'''

import tensorflow as tf
from tensorflow.keras.preprocessing import *

dsize = (128, 128)
images = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\masks"
images_map = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\maps"
images_GAN_gt = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\no_masks"

def normalize_seg(immagine): # Normalization between 0 and 1
    immagine = immagine / 255.0
    return immagine

def normalize_GAN(immagine): # Normalization between -1 e 1
    immagine = (immagine / 127.5) - 1
    return immagine

def prepare_tf_segmentation():
    X_train = image_dataset_from_directory(images, image_size=dsize, color_mode="grayscale", label_mode=None, validation_split=0.2, subset="training", shuffle=True, seed = 1, interpolation="lanczos5")
    X_val = image_dataset_from_directory(images, image_size=dsize, color_mode="grayscale", label_mode=None, validation_split=0.2, subset="validation", shuffle=True, seed = 1, interpolation="lanczos5")
    Y_train = image_dataset_from_directory(images_map, image_size=dsize, color_mode="grayscale", label_mode=None, validation_split=0.2, subset="training",shuffle=True, seed = 1, interpolation="lanczos5")
    Y_val = image_dataset_from_directory(images_map, image_size=dsize, color_mode="grayscale", label_mode=None, validation_split=0.2, subset="validation", shuffle=True, seed = 1, interpolation="lanczos5")
    X_train = X_train.map(normalize_seg)
    Y_train = Y_train.map(normalize_seg)
    dataset = tf.data.Dataset.zip((X_train , Y_train))
    X_val = X_val.map(normalize_seg)
    Y_val = Y_val.map(normalize_seg)
    dataval = tf.data.Dataset.zip((X_val , Y_val))
    return dataset, dataval

def prepare_tf_GAN():
    mask_ds = image_dataset_from_directory(images, image_size=dsize, label_mode=None, validation_split=0.05, subset="training", shuffle=True, seed = 1, interpolation="lanczos5", batch_size=16)
    mask_ds_test = image_dataset_from_directory(images, image_size=dsize, label_mode=None, validation_split=0.05, subset="validation", shuffle=True, seed = 47, interpolation="lanczos5", batch_size=16)
    map_ds = image_dataset_from_directory(images_map, image_size=dsize, color_mode="grayscale", label_mode=None, validation_split=0.05, subset="training",shuffle=True, seed = 1, interpolation="lanczos5", batch_size=16)
    map_ds_test = image_dataset_from_directory(images_map, image_size=dsize, color_mode="grayscale", label_mode=None, validation_split=0.05, subset="validation", shuffle=True, seed = 47, interpolation="lanczos5", batch_size=16)
    gt_ds = image_dataset_from_directory(images_GAN_gt, image_size=dsize, label_mode=None, validation_split=0.05, subset="training",shuffle=True, seed = 1, interpolation="lanczos5", batch_size=16)
    gt_ds_test = image_dataset_from_directory(images_GAN_gt, image_size=dsize, label_mode=None, validation_split=0.05, subset="validation", shuffle=True, seed = 47, interpolation="lanczos5", batch_size=16)
    mask_ds = mask_ds.map(normalize_GAN)
    map_ds = map_ds.map(normalize_GAN)
    mask_ds_test = mask_ds_test.map(normalize_GAN)
    map_ds_test = map_ds_test.map(normalize_GAN)
    gt_ds = gt_ds.map(normalize_GAN)
    gt_ds_test = gt_ds_test.map(normalize_GAN)
    dataset = tf.data.Dataset.zip((mask_ds, map_ds, gt_ds))
    testset = tf.data.Dataset.zip((mask_ds_test, map_ds_test, gt_ds_test))
    return dataset, testset