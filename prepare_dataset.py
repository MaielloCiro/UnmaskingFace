import cv2
import numpy as np
import os
import os.path as osp


images = "drive/MyDrive/Machine Learning/MASKFACCIA/DATASETS/colab_masked"
images_gt = "drive/MyDrive/Machine Learning/MASKFACCIA/DATASETS/colab_maps"
dsize = (256, 256)
images_original = []
images_map = []


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

def name_check(img_map, img_mask):
    path1, name1 = os.path.split(img_map)
    path2, name2 = os.path.split(img_mask)
    name1 = name1[:-4]
    search = name2.find(name1)
    if search > -1:
        return True
    return False

def prepare():
    imlist = find_path(images)
    imlist_gt = find_path(images_gt)
    count = 0

    for image in imlist:
        count += 1
        if count%10:
            print(count)
        src = cv2.imread(image)
        output = cv2.resize(src, dsize, interpolation=cv2.INTER_LANCZOS4)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        norm_image = cv2.normalize(output, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        images_original.append(norm_image)
        for image_segm in imlist_gt:
            if name_check(image_segm, image):
                src_map = cv2.imread(image_segm)
                output_map = cv2.resize(src_map, dsize, interpolation=cv2.INTER_LANCZOS4)
                output_map = cv2.cvtColor(output_map, cv2.COLOR_BGR2GRAY)
                norm_image_map = cv2.normalize(output_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                images_map.append(norm_image_map)
                break

    train_images = np.asarray(images_original).reshape(-1,256,256,1)
    gt_images = np.asarray(images_map).reshape(-1,256,256,1)

    return train_images, gt_images