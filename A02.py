import numpy as np
import cv2
import gradio as gr

def read_kernel_file(filepath):
    with open(filepath) as f:
        content = f.readline()
    
    content = content.strip()
    tokens = content.split(" ")

    rowCnt = int(tokens[0])
    colCnt = int(tokens[1])

    kernel = np.array(tokens[2:], dtype=np.str_).astype(np.float64)
    kernel = kernel.reshape(rowCnt, colCnt)

    return kernel

def convolution_preprocess(image, kernel):
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # flip kernel
    kernel = cv2.flip(kernel, -1)

    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    # extra pixels that the kernel will extend out passed in the image dimension
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode="constant", constant_values=0)

    return padded_image, kernel, image_height, image_width, kernel_height, kernel_width

def do_convolution_slow(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    padded_image, kernel, image_height, image_width, kernel_height, kernel_width = convolution_preprocess(image, kernel)

    output = np.zeros((image_height, image_width), dtype=np.float64)

    # convolution the slow way
    # loop thorugh image
    for i in range(image_height):
        for j in range(image_width):
            extracted_region = padded_image[i:i + kernel_height, j:j + kernel_width]
            
            # loop through kernel
            extracted_values = 0
            for h in range(kernel_height):
                for w in range(kernel_width):
                    extracted_values += extracted_region[h, w] * kernel[h, w]
            
            output[i, j] = extracted_values
    
    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    return output

def do_convolution_fast(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    padded_image, kernel, image_height, image_width, kernel_height, kernel_width = convolution_preprocess(image, kernel)
    
    strides = (
        padded_image.strides[0], # Moves across the rows
        padded_image.strides[1], # Move down the cols
        padded_image.strides[0], # Move across kernel rows
        padded_image.strides[1]  # Move across kernel cols
    )

    # Each kernel patch in the image
    regions = np.lib.stride_tricks.as_strided(
        padded_image,
        shape=(image_height, image_width, kernel_height, kernel_width),
        strides=strides
    )

    # Multiple each kernel patch by the kernel then sum
    output = np.sum(regions * kernel, axis=(2,3))

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    return output

def do_convolution_fourier(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    # extra pixels that the kernel will extend out passed in the image dimension
    dft_height = cv2.getOptimalDFTSize(image_height + kernel_height - 1)
    dft_width = cv2.getOptimalDFTSize(image_width + kernel_width - 1)

    padded_image = cv2.copyMakeBorder(image, 0, dft_height - image_height, 0, dft_width - image_width, cv2.BORDER_CONSTANT, value=0)
    
    padded_kernel = cv2.copyMakeBorder(kernel, 0, dft_height - kernel_height, 0, dft_width - kernel_width, cv2.BORDER_CONSTANT, value=0)

    # convert to frequency domain
    image_fft = np.fft.fft2(padded_image)
    kernel_fft = np.fft.fft2(padded_kernel)

    output_fft = image_fft * kernel_fft

    # convert to spatial domain
    output_conv = np.fft.ifft2(output_fft).real

    start_height = (kernel_height) // 2
    start_width = (kernel_width) // 2

    output = output_conv[start_height : start_height + image_height, start_width : start_width + image_width]

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    return output

def check_if_separable(kernel):
    kernel = kernel.astype(np.float64)
    U, S, VT = np.linalg.svd(kernel)
    EPS = 1e-5

    separable = (np.sum (S > EPS) == 1)

    if separable:
        u = U[:, 0]
        v = VT[0, :]

        vert_filter = (u * np.sqrt(S[0])).reshape(-1, 1)
        horiz_filter = (v * np.sqrt(S[0])).reshape(1, -1)

        return True, vert_filter, horiz_filter
    else:
        return False, None, None

def do_convolution_separable(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True, conv_func=do_convolution_fast):
    seperable, vert_filter, horiz_filter = check_if_separable(kernel)

    if seperable:
        vert_image = conv_func(image, vert_filter, alpha, beta, convert_uint8=False)
        image = conv_func(vert_image, horiz_filter, alpha, beta, convert_uint8)

        return image
    else:
        return None

def do_convolution_optimal(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    
    if kernel.shape[0] * kernel.shape[1] > 100:
        output = do_convolution_fourier(image, kernel, alpha, beta, convert_uint8)
        return output
    
    output = do_convolution_separable(image, kernel, alpha, beta, convert_uint8)

    if output is not None:
        return output
    
    else:
        output = do_convolution_fourier(image, kernel, alpha, beta, convert_uint8)
        return output

def process_gradio(input_image, input_kernel, alpha, beta):

    kernel = read_kernel_file(input_kernel.name)

    output = do_convolution_optimal(input_image, kernel, alpha, beta)

    return output

def launch_gradio():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", image_mode="L")
                input_kernel = gr.File(label="Kernel Image", file_types=[".txt"])

                with gr.Row():
                    alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="Alpha")
                    beta = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Beta")

                submit_button = gr.Button("Submit")

            with gr.Column():
                output_image = gr.Image(label="Output Image", image_mode="L")

        submit_button.click(
            fn=process_gradio,
            inputs=[input_image, input_kernel, alpha, beta],
            outputs=output_image
        )
    
    interface.launch()

def main():
    launch_gradio()

if __name__ == "__main__":
    main()