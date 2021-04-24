import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from GAN_model import *
import matplotlib.pyplot as plt

dsize = (128, 128, 3)
dmap = (128, 128, 1)

gen = generatore()
gen.summary()
path = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\dataset_masked\\149882_surgical.jpg"
Imask = tf.keras.preprocessing.image.load_img(path, target_size=dsize)

path1 = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\dataset_maps\\149882.jpg"
Imask_map = tf.keras.preprocessing.image.load_img(path1, color_mode='grayscale', target_size=dmap)
Imask = np.asarray(Imask).reshape((1, 128, 128, 3))
Imask_map = np.asarray(Imask_map).reshape((1, 128, 128, 1))
generated_image = gen([Imask, Imask_map], training=False)

plt.imshow(generated_image[0, ...])
plt.show()