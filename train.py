import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_dataset import *
from map_module_model import *

model = modello()
x_train, x_gt = prepare()
print("Preparati i dati")
history = model.fit(x_train, x_gt, epochs=100, batch_size=32)
model.save('prova.h5')

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('accuracy and loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()
