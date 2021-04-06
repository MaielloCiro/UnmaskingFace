import tensorflow as tf
import matplotlib.pyplot as plt
# from prepare_dataset import *
from map_module_model import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

model = modello()
x_train = np.load("C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\train_images.npy")
x_gt = np.load("C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\gt_images.npy")
print("Caricati i dati")
history = model.fit(x_train, x_gt, validation_split= 0.2, epochs=5, batch_size=32)
model.save('prova.h5')

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