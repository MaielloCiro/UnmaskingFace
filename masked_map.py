import numpy as np
import cv2
import os
import os.path as osp


images = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\img_align_celeba"
images_masked = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\img_align_celeba_masked"
counter = 0
counter_elimination = 0

def binarization(n, map):
    for i in range(0, 218):
        for j in range(0, 178):
            if map[i, j]<n:
                map[i, j] = 0
            else:
                map[i, j] = 255

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

def name_check(img, img_mask):
    path1, name1 = os.path.split(img)
    path2, name2 = os.path.split(img_mask)
    name1 = name1[:-4]
    search = name2.find(name1)
    if search > -1:
        return True, name1
    print(name1+" non trovata")
    return False, name1

imlist = find_path(images)
imlist2 = find_path(images_masked)

for image in imlist:
    image_masked = imlist2[counter]
    flag, name = name_check(image, image_masked)
    if flag:
        counter += 1
        print(counter)
        immagine = cv2.imread(image)
        #cv2.imshow('originale', immagine)
        immagine_masked = cv2.imread(image_masked)
        #cv2.imshow('Masked', immagine_masked)
        imMap = cv2.absdiff(immagine_masked, immagine)
        imMap = cv2.cvtColor(imMap, cv2.COLOR_BGR2GRAY)
        imMap = np.array(imMap)
        binarization(6, imMap)
        #cv2.imshow('Map with noise', imMap)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        openingMap = cv2.morphologyEx(imMap, cv2.MORPH_OPEN, kernel)
        cv2.imwrite('Maps\\'+name+'.jpg', openingMap)
        #cv2.imshow('Erosion and dilation', openingMap)
        #cv2.waitKey(0)
    else:
        os.remove(image)
        print("Rimossa!")
        counter_elimination += 1

print("Elaborate "+str(counter)+" foto")
print("Eliminate "+str(counter_elimination)+" foto")