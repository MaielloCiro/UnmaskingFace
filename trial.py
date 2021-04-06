import numpy as np
from numpy import genfromtxt

my_data1 = np.load("C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\train_images1.npy")
my_data2 = np.load("C:\\Users\\user\\Documents\\GitHub\\UnmaskingFace\\train_images2.npy")

# print(my_data)

print(my_data1.shape)
print(my_data2.shape)
c= np.concatenate((my_data1, my_data2), axis=0)
print(c.shape)