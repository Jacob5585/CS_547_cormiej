import numpy as np
import gradio as gr
import cv2

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
    # print(lut)

    return lut

def get_piecewise_linear_transform(points):
    # print(points)
    points = sorted(points, key=lambda x: x[0])

    r, s = zip(*points)
    _ = np.arange(256) # 0-255

    lut = np.interp(_, r, s)

    lut = standarize_look_up_table(lut)

    return lut

def apply_intensity_transform(image, int_transform):
    transform_image = int_transform[image]
    
    return transform_image

def estimate_gamma_exponent(image, output):
    # normalize
    # image = image.as_type(np.float64) / 255
    # output = output.as_type(np.float64) / 255
    
    image = image.ravel()#.astype(np.float64)
    output = output.ravel()#.astype(np.float64)

    historgram = np.bincount(image, minlength=256)
    sums = np.bincount(image, weights=output, minlength=256)

    lut = np.divide(sums, historgram, out=np.zeros_like(sums), where= historgram != 0)

    intensities = np.arange(256)

    masks = (historgram > 0) & (intensities > 0) & (intensities < 255) & (lut > 0)

    valid_inputs = intensities[masks] / 255
    valid_outputs = lut[masks] / 255

    gamma = np.log(valid_outputs) / np.log(valid_inputs)

    print(gamma)

    return np.mean(gamma)

def do_something_place_holder(input_image, task, stretching, gamma, max_r):
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
        
    
    return output_image

def launch_gradio():
    # Maybe only try to display them when the associated task is selcted

    with gr.Blocks() as interface:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image")
                
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
                output_image = gr.Image(label="Output Image")

        # Remove for live update
        # button.click(
        #     fn=do_something_place_holder,
        #     inputs=[input_image, tasks, stretching, gamma, max_r],
        #     outputs=output_image
        # )

        inputs = [input_image, task, stretching, gamma, max_r]

        # updates when the non assicated checkbox/slider is adjsuted <maybe lock the non assicated ones)
        for input in inputs:
            input.change(
                fn=do_something_place_holder,
                inputs=[input_image, task, stretching, gamma, max_r],
                outputs=output_image
            )
    
    interface.launch()

def main():
    get_log_transform(10)
    get_gamma_transform(10)
    
    image = np.array([[1, 0, 2], [0, 0, 0], [1, 2, 2], [0, 4, 0]], dtype="uint8")
    lut = get_hist_equalize_transform(image, False)
    # print(lut)

    points = [[0,0], [50,20], [100,200], [255,255]]
    piecewise_lut = get_piecewise_linear_transform(points)
    # print(f"piecewise_lut:\n{piecewise_lut}")

    print("\n\n")
    estimate_gamma_exponent(image, image)

    launch_gradio()

if __name__ == "__main__":
    main()