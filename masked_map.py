'''
Script to create and filter synthetic maps
We also delete original images which were wrongly masked by MaskTheFace tool
'''

import numpy as np
import cv2
import os
import os.path as osp

images = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\no_masks"
images_masked = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\masks"
counter = 0 # Counter used for masked images
counter_elimination = 0 # Counter used for number of original images eliminated

'''
Function to create a list of directory entries
'''
def find_path(im):
    try:
        imlist = [osp.join(osp.realpath('.'), im, img) for img in os.listdir(im) if os.path.splitext(img)[1] ==
                  '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] == '.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), im))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(im))
        exit()
    return imlist

'''
Function to check if for each original image 
there is its counterpart masked
'''
def name_check(img, img_mask):
    path1, name1 = os.path.split(img)
    path2, name2 = os.path.split(img_mask)
    name1 = name1[:-4]
    search = name2.find(name1)
    if search > -1:
        return True, name1
    print(name1+" non trovata")
    return False, name1

'''
Function to eliminate intermediate value 
and obtain only black and white values
'''
def binarization(n, map): # n is arbitrary chosen after several tests
    for i in range(0, 218):
        for j in range(0, 178):
            if map[i, j]<n:
                map[i, j] = 0
            else:
                map[i, j] = 255
				
def filter(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) # Filter with erosion and dilation 

imlist = find_path(images)
imlist2 = find_path(images_masked)

for image in imlist:
    image_masked = imlist2[counter]
    flag, name = name_check(image, image_masked)
    if flag: # If corresponding found
        counter += 1
        immagine = cv2.imread(image)
        #cv2.imshow('originale', immagine)
        immagine_masked = cv2.imread(image_masked)
        #cv2.imshow('Masked', immagine_masked)
        imMap = cv2.absdiff(immagine_masked, immagine) # Difference (with absolute value) between original image and corresponding masked image
        imMap = cv2.cvtColor(imMap, cv2.COLOR_BGR2GRAY)
        imMap = np.array(imMap)
        binarization(6, imMap)
        #cv2.imshow('Map with noise', imMap)
        openingMap = filter(imMap)
        cv2.imwrite('C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\maps\\'+name+'.jpg', openingMap)
        #cv2.imshow('Erosion and dilation', openingMap)
        #cv2.waitKey(0)
    else:
        os.remove(image)
        print("Rimossa!")
        counter_elimination += 1

print("Elaborate "+str(counter)+" foto")
print("Eliminate "+str(counter_elimination)+" foto")