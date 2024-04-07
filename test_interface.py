import gradio as gr
from dataloader import Dataset_h5py_test

def hello():
    print("Hello")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            dataset = gr.Dropdown(["Quickbird"], label="Dataset", type="value",)
            data_type = gr.Dropdown(["H5py"], label="Data Type", type="value")

            dataset.change(fn=hello)
            data_type.change(fn=hello)

            preview_image_index = gr.Dropdown([1,2,3], label="Preview Image Index", type="index")
            preview = gr.Image('data\\h5py\\qb\\full_examples\\data_overview_qb_full.png')

            preview_image_index.change(fn=hello)

            model = gr.Dropdown(["MSDCNN"], label="Model")
            btn = gr.Button("Go")
        with gr.Column():
            results = gr.Image()



demo.launch()