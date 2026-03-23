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
This program performs grayscale image transformations for log, gamma, equalization, and piecewize transforms.  
To estimate gamma weighted least square of log linear regression is used.  
When the program runs, the terminal will output a url for gradio. In gradio you can upload an image and choice the transformation to apply. This then will return the transformed grayscale image, the input, output histogram, and the transformation graph.

### A02.py
This program performs convolutions on grayscale images. In gradio you can upload an image, upload a kernel, select a number for alpha, and beta. Gradio will choice an optimized convoultion function, apply the kernel for convolution to the image, then return the convultion image to gradio.

#### Optimal Convolution
justify your implementation choice by providing timing charts from running the evaluation script and your own explanation/analysis text
The optimal implementation checks 

<!-- ![Chart for timings](assign02/charts/AllTimingsGraph.png)
![Chart for timings](assign02/charts/AllTimingsSumGraph.png) -->