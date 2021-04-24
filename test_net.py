import tensorflow as tf
from tensorflow.keras.preprocessing import *
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

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

def normalize(immagine):
    immagine = immagine / 255.0
    return immagine

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

images = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\testmasked"
images_gt = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\testmaps"
X_test = image_dataset_from_directory(images, image_size=(128, 128), color_mode="grayscale", label_mode=None, shuffle=False, interpolation="lanczos5")
Y_test = image_dataset_from_directory(images_gt, image_size=(128, 128), color_mode="grayscale", label_mode=None, shuffle=False, interpolation="lanczos5")
X_test = X_test.map(normalize)
Y_test = Y_test.map(normalize)
testset = tf.data.Dataset.zip((X_test , Y_test))



modello_caricato = load_model("segmentation_model_20k_20epoch.h5")
#---------  EVALUATION --------
evaluation = modello_caricato.evaluate(testset)
print("%s: %.2f%%" % (modello_caricato.metrics_names[1], evaluation[1]*100))

#---------  PREDICTION --------
prediction = modello_caricato.predict(x = X_test)
count = 0
fig = plt.figure(figsize=(10, 10))
for image in prediction:
    count += 1
    binarization(0.01, image)
    image = filter(image)
    stringa = "Risultati\\Prediction20k\\output_seg_20k_20epoch" + str(count) + ".png"
    plt.imshow(image, cmap='gray')
    fig.savefig(stringa)
    print(count)


def predictpicture():
    dsize = (128, 128)
    path = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\testmasked\\testset_masked\\039692_N95.jpg"
    image = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=dsize, interpolation="lanczos")
    numpyimage = np.array(image)
    X = numpyimage / 255.0
    X = X.reshape(1, 128, 128, 1)
    prediction = modello_caricato.predict(x = X)
    example = prediction[0, :, :, 0]
    stringa = "Risultati\\Prediction20k\\output_seg_20k_20epoch10.png"
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(numpyimage)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(example, cmap='gray')
    plt.title("Segmentation")
    plt.show()
    fig.savefig(stringa)