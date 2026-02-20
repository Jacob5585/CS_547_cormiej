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
    # cdf_min = np.min(cdf[cdf != 0])
    # cdf_max = np.max(cdf)

    # print(f"\nCDF:\n{cdf[0]}\n\n")
    # print(f"\nnormalized_historgram:\n{normalized_historgram[0]}\n\n")

    # Stretching
    if do_stretching == True:
        cdf = cdf - cdf[0]
        cdf = cdf / cdf[-1]

    lut = cdf * max_intensity
    lut = standarize_look_up_table(lut)

    return lut

def get_piecewise_linear_transform(points):
    points = sorted(points, key=lambda x: x[0])

    r, s = zip(*points)
    other_r = np.arange(256) # 0-255

    lut = np.interp(other_r, r, s)

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
    plt.bar(np.arange(256), histogram, color="gray", width=1.0)
    plt.title("Histogram")
    plt.xlim([0, 255])
    plt.tight_layout()

    return fig

def get_transformation_image(input_image, output_image):

    return fig

def process_gradio(input_image, task, stretching, gamma, max_r):
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

    # elif task == "piecewise":
    #     lut = get_piecewise_linear_transform())
    #     output_image = lut[grayscale]
    
    input_historgram = get_histogram_image(input_image)
    output_historgram = get_histogram_image(output_image)
    
    return output_image, input_historgram, output_historgram

def launch_gradio():
    # Maybe only try to display them when the associated task is selcted

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                task = gr.Radio(
                    choices=["Histogram Equalization", "Gamma", "Log"],
                    label="Options",
                    value="Histogram Equalization"
                )

                with gr.Group():
                    stretching = gr.Checkbox(label="do_stretching", value=True)
                    gamma = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Gamma Exponent")
                    max_r = gr.Slider(1, 255, value=255, step=1, label="Max_r")

                # button = gr.Button() Remove for live update

            with gr.Column():
                input_image = gr.Image(label="Input Image")
                input_historgram = gr.Plot(label="Output Histogram")
                

            with gr.Column():
                output_image = gr.Image(label="Output Image")
                output_historgram = gr.Plot(label="Output Histogram")


        # Remove for live update
        # button.click(
        #     fn=process_gradio,
        #     inputs=[input_image, tasks, stretching, gamma, max_r],
        #     outputs=output_image
        # )

        inputs = [input_image, task, stretching, gamma, max_r]
        outputs = [output_image, input_historgram, output_historgram]

        # updates when the non assicated checkbox/slider is adjsuted <maybe lock the non assicated ones)
        for input in inputs:
            input.change(
                fn=process_gradio,
                inputs=[input_image, task, stretching, gamma, max_r],
                outputs=outputs
            )

    interface.launch()

def main():
    get_log_transform(10)
    get_gamma_transform(10)
    
    image = np.array([[1, 0],[2, 4]], dtype="uint8")
    output = np.array([[1, 1], [3, 5]], dtype="uint8")
    lut = get_hist_equalize_transform(image, False)
    # print(lut)

    points = [[0,0], [50,20], [100,200], [255,255]]
    piecewise_lut = get_piecewise_linear_transform(points)
    # print(f"piecewise_lut:\n{piecewise_lut}")

    print("\n\n")
    gamma = estimate_gamma_exponent(image, output)
    print(gamma)

    launch_gradio()

if __name__ == "__main__":
    main()