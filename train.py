import tensorflow as tf
from prepare_dataset import *
from map_module_model import *

model = modello()
x_train, x_gt = prepare()
print("Preparati i dati")
model.fit(x_train, x_gt, epochs=100)
model.save('prova.h5')
