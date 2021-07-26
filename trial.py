import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from GAN_model import *
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing import *
import os.path as osp

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

def normalize_seg(immagine):
    immagine = immagine / 255.0
    return immagine

def find_path(im):
    try:
        imlist = [osp.join(osp.realpath('.'), im, img) for img in os.listdir(im) if os.path.splitext(img)[1] ==
                  '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), im))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(im))
        exit()
    return imlist

count = 0
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

def map_generation(images, names):
    prediction = seg_model.predict(x = images)
    count = 0
    stringa = 'C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_MAP\\test\\'
    for image in prediction:
        binarization(0.01, image)
        image = filter(image)
        image = image.reshape(128,128,1)
        tf.keras.preprocessing.image.save_img(stringa+names[count]+'.jpg', image)
        count+=1
    # example = np.asarray(example)
    # example = example.reshape(128, 128, 1)
    # stringa = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST\\145997_surgical_segmentation.jpg"
    # fig = plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(numpyimage)
    # plt.title("Original")
    # plt.subplot(1, 2, 2)
    # plt.imshow(example, cmap='gray')
    # plt.title("Segmentation")
    # fig.savefig(stringa)    

def generate_image(model, test_input, test_map, names):
    global count
    prediction = model([test_input, test_map], training=True)
    fig=plt.figure(figsize=(15, 7.5))
    plt.suptitle("TEST", fontsize=24, ha="center")
    display_list = [test_input[0, :, :, :], test_map[0, :, :, :], prediction[0, :, :, :]]
    title = ['Test Input', 'Segmentation Map', 'Predicted Image']
    stringa ="C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_OUTPUT\\"
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i], fontsize=18)
        # getting the pixel values between [0, 1] to plot it.
        if i == 1:
            plt.imshow(display_list[i], cmap = 'gray')
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    fig.savefig(stringa+names[count]+'.jpg')
    plt.close('all')
    count+=1

dsize = (128, 128)
path = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\testmasked1k"
mask = image_dataset_from_directory(path, image_size=(128, 128), color_mode='grayscale', label_mode=None, shuffle=False, interpolation="lanczos5", batch_size=16)
imlist = find_path(path+"\\testsetmasked")
names=[]
for i in imlist:
    i = i.split("\\")
    name = i[-1]
    name = name.split('_')
    name = name[0]
    names.append(name)
mask_seg = mask.map(normalize_seg)
map_generation(mask_seg, names)
mask = image_dataset_from_directory(path, image_size=(128, 128), label_mode=None, shuffle=False, interpolation="lanczos5", batch_size=16)
mask_GAN = mask.map(normalize_GAN)
pathmap = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\TEST_MAP"
map = image_dataset_from_directory(pathmap, image_size=(128, 128), color_mode='grayscale', label_mode=None, shuffle=False, interpolation="lanczos5", batch_size=16)
map = map.map(normalize_GAN)
testset = tf.data.Dataset.zip((mask_GAN,map))
for n, (example_input, example_map) in testset.enumerate():
    generate_image(gen, example_input, example_map, names)
