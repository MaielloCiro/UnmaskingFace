# Unmasking Face

## Why
This project was developed for a Machine Learning course attended at [Politecninco Di Torino](http://www.polito.it). Because of the increased use of masks in the last year due to the actual pandemic situation, we chose a project that consists in removing masks and generating the hidden part of the face (image inpainting task).

## Who
It was developed by Mattia Bencivenga, Ciro Maiello and Ivan Orefice.

## What
Inspired by [A Novel GAN-Based Network for Unmasking of Masked Face](https://ieeexplore.ieee.org/abstract/document/9019697), we developed a model divided in two main modules:
- Map Module, a modified version of the U-Net, whose purpose is to create a segmentation map of the mask;
- Editing Module, composed of a GAN (U-Net for the generator, Pix2Pix for the two discriminators) and a Perceptual Network, whose purpose is to generate the face under the mask;


## Results
You can see the complete report in the file "Elaborato.pdf". It is written based on a template, given by our professors, used in CVPR conferences. There we explained all the architectural choices that we made with their motivation.
Here we'll report some results.
### Examples
![Esempio1](https://github.com/MaielloCiro/UnmaskingFace/blob/main/Risultati/039733.png "Esempio 1")
![Esempio2](https://github.com/MaielloCiro/UnmaskingFace/blob/main/Risultati/039740.png "Esempio 2")
![Esempio3](https://github.com/MaielloCiro/UnmaskingFace/blob/main/Risultati/039743.png "Esempio 3")

As you can see, the results can be considered more than acceptable even taking into account that our resources were limited and our dataset, created by ourselves, was too homogeneous.
As a matter of fact, the model is affected by overfitting and it does not generalize properly.
This problem could have been solved making the dataset more heterogeneous.

## Try me
In order to test the model, execute the file [trial.py](https://github.com/MaielloCiro/UnmaskingFace/blob/main/trial.py). This file will take all the required images the model needs (masked images) from the directory [TestSet/testmasked1k](https://github.com/MaielloCiro/UnmaskingFace/tree/main/TestSet/testmasked1k); firstly, it will generate the maps from the masked images and will save them in [TestSet/test_map](https://github.com/MaielloCiro/UnmaskingFace/tree/main/TestSet/test_map); afterwards, it uses these maps for the editing module, and it generates the unmasked images, saved in [TestSet/Results](https://github.com/MaielloCiro/UnmaskingFace/tree/main/TestSet/Results). 
