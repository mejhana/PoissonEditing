# Poisson Editing
Poisson Editing is a image editing tool that blends two images together in a seamless manner. My implementation is based on the paper [poisson image editing](https://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf). The method is based on solving a Poisson equation, which allows for the transfer of gradients between the two images while preserving the overall structure of the target image.

# How to use 
1. Take 2 images source and destination and place it in "images" directory. 
2. Align the source image in the desired location on the destination image and fill the rest of the pixels with black such that source and destination images are of same size
3. For the aligned and filled source image, get a ROI using any [image annotating tool](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html). 
4. If you are using VGG annotation tool, export the annotation as a json file (annotations -> export annotations as json), place this file in the project folder. Rename this file as "annotation.json"
5. Run main.py, change "clone" to False, if you want to implement fill and set "mixGrad" to True if you want to use mixture of gradients

