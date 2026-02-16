import numpy as np
import cv2
import matplotlib.pyplot as plt
import gradio as gr

def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                color_image = gr.Image(label="Input Image")
            with gr.Column():
                gray_image = gr.Image(label="Grayscale Image")
                one_button = gr.Button("MAGIC")

    def do_magic(gray):
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        gray = np.where(gray > 200, 255, 0)
        return gray

    # one_button.click()
    
    def on_upload(color):
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return gray
    
    color_image.upload(fn=on_upload, inputs=color_image, outputs=gray_image)
    
    demo.launch()

if __name__ == "__main__":
    main()