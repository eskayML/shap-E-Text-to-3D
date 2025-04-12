#!/usr/bin/env python

import gradio as gr
import torch

from app_image_to_3d import create_demo as create_demo_image_to_3d
from app_text_to_3d import create_demo as create_demo_text_to_3d
from model import Model

DESCRIPTION = """
![nosana banner](https://gcdnb.pbrd.co/images/0juVIv58VlRy.png)
# Nosana 3D Generation Demo
[Link to model](https://github.com/openai/shap-e)

"""

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

model = Model()

with gr.Blocks(css_paths="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.Tab(label="Text to 3D"):
            create_demo_text_to_3d(model)
        with gr.Tab(label="Image to 3D"):
            create_demo_image_to_3d(model)

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
