import tensorflow as tf
import matplotlib.pyplot as plt
from map_module_model import *
from prepare_dataset import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

model = modello()
dataset, dataval = prepare_tf()
print("Caricati i dati")

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\Checkpoints\\cp{epoch:08d}.h5", save_weights_only=False, save_freq=5*500)
history = model.fit(dataset, validation_data=dataval, epochs=20) #,callbacks=[cp_callback]
model.save('segmentation_model_20k_20epoch.h5')

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

# x_test = np.load("C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\test_images.npy")
# x_testgt = np.load("C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\testgt_images.npy")
# evaluation = model.evaluate(x_test, x_testgt)
# print("%s: %.2f%%" % (model.metrics_names[1], evaluation[1]*100))

# prediction = model.predict(x = x_test[0].reshape(1, 128, 128, 1)) #ci deve annà x_test
# cv2.imshow("Output", prediction)
# cv2.waitKey(0)
