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
