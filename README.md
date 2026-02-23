# CS 547: Computer Vision and Image Processing
***Spring 2026***  
***Author: Jacob Cormier***  
***Original Author: Dr. Michael J. Reale***  
***SUNY Polytechnic Institute*** 

## Runnable Python Scripts

### BasicVision.py
A basic sample that loads up the relevant libraries, prints versions numbers, and either 1) loads an image from a path specified on the command line, or 2) opens a webcam.
Image(s) will be displayed until a key is hit.

### A01.py
This program performs grayscale iamge transformations for log, gamma, equalization, and piecewize transforms.  
To estimate gamma the least square derivation of linear regression is used.  
When the program runs, the terminal will output a url for gradio. In gradio you can upload an image and choice the transformation to apply. This then will return the transformed grayscale image, the input, output histogram, and the transformation graph.