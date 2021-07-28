'''
Training Map Module
'''

import tensorflow as tf
import matplotlib.pyplot as plt
from map_module_model import *
from prepare_dataset import *

'''
Code to avoid OoM
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

'''
Loading dataset and training
'''
model = seg_model()
dataset, dataval = prepare_tf_segmentation()
print("Caricati i dati")

history = model.fit(dataset, validation_data=dataval, epochs=20)
model.save('segmentation_model_20k_20epoch.h5')

'''
Plotting Loss and Accuracy
'''
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
