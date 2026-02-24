import numpy as np
import gradio as gr
import cv2
import matplotlib.pyplot as plt

def standarize_look_up_table(lut):
    lut = np.round(lut)
    lut = np.clip(lut, a_min=0, a_max=255)
    lut = lut.astype("uint8")

    return lut

def get_log_transform(max_r):
    r = np.arange(256) # 0-255
    c = (255 / np.log(1 + max_r))
    s = c * np.log(1 + r) #s is a numpy array

    lut = standarize_look_up_table(s)
   
    return lut

def get_gamma_transform(gamma):
    r = np.arange(256) # 0-255
    s = 255 * ((r / 255) ** gamma) #s is a numpy array

    lut = standarize_look_up_table(s)

    return lut

def get_hist_equalize_transform(image, do_stretching):
    image = image.ravel()
    max_intensity = 255
    histogram = np.bincount(image, minlength=256)
    normalized_histogram = histogram / image.size

    # cdf = cumulative distribution function
    cdf = np.cumsum(normalized_histogram)

    if do_stretching:
        cdf = cdf - cdf[0]
        cdf = cdf / cdf[-1]

    lut = cdf * max_intensity
    lut = standarize_look_up_table(lut)

    return lut

def get_piecewise_linear_transform(points):
    points = sorted(points, key=lambda x: x[0])

    r_knots, s_knots = zip(*points)
    r = np.arange(256) # 0-255

    # apply linear interpolation
    lut = np.interp(r, r_knots, s_knots)

    lut = standarize_look_up_table(lut)

    return lut

def apply_intensity_transform(image, int_transform):
    lut = int_transform[image]
    
    return lut

def estimate_gamma_exponent(image, output):
    image = image.ravel()
    output = output.ravel()

    # Map image pixel value to output
    transform = np.zeros((256), dtype="uint8")
    transform[image]= output

    # Track which vlaues in image have a mapping
    valid_mask = np.zeros((256), dtype="bool")
    valid_mask[image]= True

    # Get the pixel values where mask is true
    valid_indices = np.arange(256, dtype="uint8")[valid_mask]

    # Get the output values where mask is true
    valid_transform_vals = transform[valid_mask]

    # removes 0 as /0 is undefined
    zero_mask = (valid_indices > 0) & (valid_transform_vals > 0)
    valid_indices = valid_indices[zero_mask].astype(np.float64)
    valid_transform_vals = valid_transform_vals[zero_mask].astype(np.float64)
    
    # Linearize and normalize the exponential
    log_r = np.log(valid_indices / 255.0) # input
    log_s = np.log(valid_transform_vals / 255.0) # output
    
    scaled_valid_indices = valid_indices ** 2

    # Weighted Least Square of Log Linear Regression
    # Weighted covariance between log_r and log_s divided  weigted variance of log_r
    # calcualtes the slope of the log linear regression
    gamma = (np.sum(log_r * log_s * scaled_valid_indices)) / (np.sum(log_r * log_r * scaled_valid_indices))

    return gamma

def get_histogram_image(image):
    histogram = np.bincount(image.ravel(), minlength=256)

    fig = plt.figure(figsize=(4, 4))
    plt.title("Histogram")
    plt.bar(np.arange(256), histogram, color="gray", width=1.0)
    plt.xlim([0, 255])
    plt.tight_layout()

    return fig

def get_transformation_image(lut):
    x = np.arange(256)

    fig = plt.figure(figsize=(4, 4))
    plt.title("Transformation Function")
    plt.xlabel("Input Intensity")
    plt.ylabel("Ouput Intensity")
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.plot(x, lut, color="gray", linewidth=2)

    plt.tight_layout()    

    return fig

def process_gradio(input_image, task, stretching, gamma, max_r, points_type):

    if input_image is None:
        return None, None, None, None

    # gradio takes image in as RGB
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    if task == "Histogram Equalization":
        lut = get_hist_equalize_transform(grayscale, stretching)
        # output_image = lut[grayscale]
        output_image = apply_intensity_transform(grayscale, lut)

    elif task == "Gamma":
        lut = get_gamma_transform(gamma)
        # output_image = lut[grayscale]
        output_image = apply_intensity_transform(grayscale, lut)

    elif task == "Log":
        lut = get_log_transform(max_r)
        # output_image = lut[grayscale]
        output_image = apply_intensity_transform(grayscale, lut)

    elif task == "Piecewise":
        if points_type == "Contrast":
            points = [[0,0], [50,20], [100,200], [255,255]]

        elif points_type == "Slicing":
            points = [[0,10], [100,10], [101,200], [200,200], [201,10], [255,10]]

        lut = get_piecewise_linear_transform(points)
        # output_image = lut[grayscale]
        output_image = apply_intensity_transform(grayscale, lut)
    
    input_histogram = get_histogram_image(grayscale)
    output_histogram = get_histogram_image(output_image)
    transformation_plot = get_transformation_image(lut)
    
    return output_image, input_histogram, output_histogram, transformation_plot

def unlock_input(task):

    # return true on the active task
    return (
        gr.update(interactive=(task == "Histogram Equalization")),
        gr.update(interactive=(task == "Gamma")),
        gr.update(interactive=(task == "Log")),
        gr.update(interactive=(task == "Piecewise"))
    )

def launch_gradio():
    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                task = gr.Radio(
                    choices=["Histogram Equalization", "Gamma", "Log", "Piecewise"],
                    label="Options",
                    value="Histogram Equalization"
                )

                with gr.Group():
                    stretching = gr.Checkbox(label="do_stretching", value=True, interactive=True)
                    gamma = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Gamma Exponent", interactive=False)
                    max_r = gr.Slider(1, 255, value=255, step=1, label="Max_r", interactive=False)
                    points = gr.Radio(
                        choices=["Contrast", "Slicing"],
                        label="Points",
                        value="Contrast", 
                        interactive=False
                    )

                transformation_plot = gr.Plot(label="Transformation Plot")

            with gr.Column():
                input_image = gr.Image(label="Input Image")
                input_histogram = gr.Plot(label="Output Histogram")

            with gr.Column():
                output_image = gr.Image(label="Output Image")
                output_histogram = gr.Plot(label="Output Histogram")

        inputs = [input_image, task, stretching, gamma, max_r, points]
        outputs = [output_image, input_histogram, output_histogram, transformation_plot]

        task.change(
            fn=unlock_input,
            inputs=[task],
            outputs=[stretching, gamma, max_r, points]
        )

        # updates when the non assicated checkbox/slider is adjsuted <maybe lock the non assicated ones)
        for input in inputs:
            input.change(
                fn=process_gradio,
                inputs=[input_image, task, stretching, gamma, max_r, points],
                outputs=outputs
            )

    interface.launch()

def main():
    launch_gradio()

if __name__ == "__main__":
    main()