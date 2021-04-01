import os
import os.path as osp

#images = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\img_align_celeba"
images_masked = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\dataset_masked"
images_segmentation = "C:\\Users\\user\\Documents\\PoliTo\\2 anno 1 semestre\\Machine learning for vision and multimedia\\PROGETTO\\DATASET\\dataset_maps"
counter_mask = 0
#counter_segment = 0
counter_elimination = 0

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

#imlist = find_path(images)
imlist_mask = find_path(images_masked)
imlist_segmentation = find_path(images_segmentation)

for image in imlist_segmentation:
    image_masked = imlist_mask[counter_mask]
    #image_segment = imlist_segmentation[counter_segment]
    flag, name = name_check(image, image_masked)
    if flag:
        counter_mask += 1
    else:
        os.remove(image)
        #os.remove(image_segment)
        print(name + "Rimossa!")
        counter_elimination += 1
    #counter_segment += 1

print("Elaborate "+str(counter_mask)+" foto")
print("Eliminate "+str(counter_elimination)+" foto")