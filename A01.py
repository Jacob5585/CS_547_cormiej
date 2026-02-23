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
    # max_intensity = np.max(image)
    max_intensity = 255
    historgram = np.bincount(image, minlength=256)
    normalized_historgram = historgram / image.size

    cdf = np.cumsum(normalized_historgram)

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

    lut = np.interp(r, r_knots, s_knots)

    lut = standarize_look_up_table(lut)

    return lut

def apply_intensity_transform(image, int_transform):
    transform_image = int_transform[image]
    
    return transform_image

def estimate_gamma_exponent(image, output):
    image = image.ravel()#.astype(np.float64)
    output = output.ravel()#.astype(np.float64)

    transform = np.zeros((256), dtype="uint8")
    transform[image]= output
    valid_mask = np.zeros((256), dtype="bool")
    valid_mask[image]= True
    valid_indices = np.arange(256, dtype="uint8")[valid_mask]
    valid_transform_vals = transform[valid_mask]

    # removes 0 as /0 is undefined
    zero_mask = (valid_indices > 0) & (valid_transform_vals > 0)
    valid_indices = valid_indices[zero_mask].astype(np.float64)
    valid_transform_vals = valid_transform_vals[zero_mask].astype(np.float64)
    
    # Least square derivation of linear regression
    log_r = np.log(valid_indices)
    log_s = np.log(valid_transform_vals)
    mean_r = np.mean(log_r)
    mean_s = np.mean(log_s)

    gamma = np.sum((log_r - mean_r)*(log_s - mean_s)) / np.sum((log_r - mean_r)**2)

    return gamma

def get_histogram_image(image):
    histogram = np.bincount(image.ravel(), minlength=256)

    fig = plt.figure(figsize=(4,3))
    plt.title("Histogram")
    plt.bar(np.arange(256), histogram, color="gray", width=1.0)
    plt.xlim([0, 255])
    plt.tight_layout()

    return fig

def get_transformation_image(lut):
    x = np.arange(256)

    fig = plt.figure(figsize=(4,3))
    plt.title("Transformation Function")
    plt.xlabel("Input Intensity")
    plt.ylabel("Ouput Intensity")
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.plot(x, lut, color="gray", linewidth=2)

    plt.tight_layout()    

    return fig

def process_gradio(input_image, task, stretching, gamma, max_r, points_type):
    # gradio takes image in as RGB
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

    if task == "Histogram Equalization":
        lut = get_hist_equalize_transform(grayscale, stretching)
        output_image = lut[grayscale]

    elif task == "Gamma":
        lut = get_gamma_transform(gamma)
        output_image = lut[grayscale]

    elif task == "Log":
        lut = get_log_transform(max_r)
        output_image = lut[grayscale]

    elif task == "Piecewise":
        if points_type == "Contrast":
            points = [[0,0], [50,20], [100,200], [255,255]]
        elif points_type == "Sliceing":
            points = [[0,10], [100,10], [101,200], [200,200], [201,10], [255,10]]

        lut = get_piecewise_linear_transform(points)
        output_image = lut[grayscale]
        print(f"\n\n\n\noutput_image: {output_image}\n\n\n\n")
    
    input_historgram = get_histogram_image(input_image)
    output_historgram = get_histogram_image(output_image)
    transformation_plot = get_transformation_image(lut)
    
    return output_image, input_historgram, output_historgram, transformation_plot

def unlock_input(task):

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
                        choices=["Contrast", "Sliceing"],
                        label="Points",
                        value="Contrast", 
                        interactive=False
                    )

                transformation_plot = gr.Plot(label="Output Histogram")

            with gr.Column():
                input_image = gr.Image(label="Input Image")
                input_historgram = gr.Plot(label="Output Histogram")

            with gr.Column():
                output_image = gr.Image(label="Output Image")
                output_historgram = gr.Plot(label="Output Histogram")

        inputs = [input_image, task, stretching, gamma, max_r, points]
        outputs = [output_image, input_historgram, output_historgram, transformation_plot]

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