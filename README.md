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

##### General Functionality
This program performs convolutions on grayscale images. In gradio you can upload an image, upload a kernel, select a number for alpha, and beta. Gradio will choose an optimized convolution function, apply the kernel for convolution to the image, then return the convultion image to gradio.

##### Optimal Convolution
The optimal implementation first checks if the kernel size exceeds 100, if so it uses Fourier convolution. If not attempts to apply separable fast convolution. When the kernel is not separable, it falls back to Fourier convolution.
For odd kernels separable  is not possible to use the separable approch, so Fourier is the only method used, it outperforms the fast method no matter the kernel size.
For Gaussian kernels, the separable fast method outperforms the Fourier convolution for smaller kernel sizes (kenrel < 11x11), so when a kernel size > 100 is reached it switches to using the Fourier method.
Based on the results, for odd kernels, the optimal method performs on par with Fourier convolution and outperforms the fast method. The overhead of switching to the frequency domain using FFT is lower than the cost of executing fast convolution in these cases.
For Gaussian kernels, the optimal approach outperforms all other methods (fast, Fourier, separable fast, and separable Fourier). Separable convolution reduces the effective dimensionality of the kernel, significantly speeding up computation. However, for sufficiently large kernels, FFT-based convolution becomes more efficient, making the switch to Fourier convolution the best choice.

![Chart for timings](assign02/output/AllTimingsGraph.png) ![Chart for timings](assign02/output/AllTimingsSumGraph.png)