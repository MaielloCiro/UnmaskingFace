import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_dataset import *
from map_module_model import *

model = modello()
x_train, x_gt = prepare()
print("Preparati i dati")
history = model.fit(x_train, x_gt, epochs=5, batch_size=16)
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
