import numpy as np
import cv2

def read_kernel_file(filepath):
    with open(filepath) as f:
        content = f.read()
    
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

    # print(f"image.shape:\n{image.shape}")
    # print(f"kernel.shape:\n{kernel.shape}")

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
            
            # print(kernel_extract)

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
    pass

def check_if_seperable(kernel):
    pass

def do_convolution_separable(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True, conv_func=do_convolution_fast):
    pass

def do_convolution_optimal(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    pass

def main():

    # for i in range(9):
    #     filepath = f'./assign02/filters/Filter_00{i}.txt'
    #     kernel = read_kernel_file(filepath)
    #     print(filepath, ":\n", kernel, "\n\n")

    filepath = f'./assign02/filters/Filter_000.txt'
    kernel = read_kernel_file(filepath)
    image = f'./assign02/images/basic00.png'
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    image_slow = do_convolution_slow(image, kernel)
    cv2.imwrite("image_slow.png", image_slow)

    image_fast = do_convolution_fast(image, kernel)
    cv2.imwrite("image_fast.png", image_fast)

    pass

if __name__ == "__main__":
    main()