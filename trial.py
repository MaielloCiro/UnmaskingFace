'''
Testing whole architecture
'''

import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from editing_module_model import *
from prepare_dataset import *
from masked_map import filter
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import *
import os.path as osp
import numpy as np

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
Function to eliminate intermediate values
and obtain only black and white values
'''
def binarization(n, map):
    for i in range(0, 128):
        for j in range(0, 128):
            if map[i, j]<n:
                map[i, j] = 0
            else:
                map[i, j] = 1

'''
Segmentation
'''
def map_generation(images, names):
    prediction = seg_model.predict(x = images)
    count = 0
    stringa = "TestSet\\test_map\\test\\"
    for image in prediction:
        binarization(0.01, image)
        image = filter(image)
        image = image.reshape(128,128,1)
        tf.keras.preprocessing.image.save_img(stringa+names[count]+'.jpg', image)
        count+=1  

'''
Generation synthetic unmasked images
'''
def generate_image(model, test_input, test_map, names):
    global count
    prediction = model([test_input, test_map], training=True)
    fig=plt.figure(figsize=(15, 7.5))
    plt.suptitle("TEST", fontsize=24, ha="center")
    display_list = [test_input[0, :, :, :], test_map[0, :, :, :], prediction[0, :, :, :]]
    title = ['Test Input', 'Segmentation Map', 'Predicted Image']
    stringa ="TestSet\\Results\\"
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i], fontsize=18)
        # Getting the pixel values between [0, 1] to plot it.
        if i == 1:
            plt.imshow(display_list[i], cmap = 'gray')
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    tf.keras.preprocessing.image.save_img(stringa+names[count]+'.jpg', prediction[0, :, :, :])
    # fig.savefig(stringa+names[count]+'.jpg')
    plt.close('all')
    count+=1

def compute_metrics(model, test_input, test_map, tar):
    prediction = model([test_input, test_map], training=True)
    score_SSIM = tf.image.ssim(prediction, tar, max_val=2.0)
    score_PSNR = tf.image.psnr(prediction, tar, max_val=2.0)
    np_score_PSNR=score_PSNR.numpy()
    average_PSNR = np.average(np_score_PSNR)
    np_score_SSIM=score_SSIM.numpy()
    average_SSIM = np.average(np_score_SSIM)
    return average_PSNR, average_SSIM

'''
Checkpoint and models restore
'''
seg_model = load_model("Checkpoints\\Segmentation\\segmentation_model_20k_20epoch.h5")
gen = generator()
checkpoint_dir = "Checkpoints\\GAN\\final_loss"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=gen)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
checkpoint.restore(ckpt_manager.latest_checkpoint)
if ckpt_manager.latest_checkpoint:
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

count = 0
names, mask_seg = prepare_tf_testseg()
map_generation(mask_seg, names)
print("Segmentation of masks: DONE")
testset = prepare_tf_testset()
for n, (example_input, example_map) in testset.enumerate():
    generate_image(gen, example_input, example_map, names)
print("Generation of synthetical images: DONE")

'''
Computation of SSIM and PSNR of testset (uncomment the two commented lines (78 and 81) in the file prepare_dataset in order to compute these metrics)
'''
# PSNR = 0
# SSIM = 0
# for n, (example_input, example_map, target) in testset.enumerate():
#     tempPSNR, tempSSIM = compute_metrics(gen, example_input, example_map, target)
#     PSNR += tempPSNR
#     SSIM += tempSSIM
# PSNR = PSNR / 63.0
# SSIM = SSIM / 63.0
# print("PSNR: " + str(PSNR))
# print("SSIM: " + str(SSIM))
