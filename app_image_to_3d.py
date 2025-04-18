#!/usr/bin/env python

import pathlib
import shlex
import subprocess

import gradio as gr
import PIL.Image
import spaces

from model import Model
from settings import MAX_SEED
from utils import randomize_seed_fn


def create_demo(model: Model) -> gr.Blocks:
    @spaces.GPU
    def run(image: PIL.Image.Image, seed: int, guidance_scale: float, num_inference_steps: int) -> str:
        return model.run_image(image, seed, guidance_scale, num_inference_steps)

    with gr.Blocks() as demo:
        with gr.Group():
            image = gr.Image(label="Input image", show_label=False, type="pil")
            run_button = gr.Button("Run")
            result = gr.Model3D(label="Result", show_label=False)
            with gr.Accordion("Advanced options", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=1,
                    maximum=20,
                    step=0.1,
                    value=3.0,
                )
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=2,
                    maximum=100,
                    step=1,
                    value=64,
                )

        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            api_name=False,
            concurrency_limit=None,
        ).then(
            fn=run,
            inputs=[
                image,
                seed,
                guidance_scale,
                num_inference_steps,
            ],
            outputs=result,
            api_name="image-to-3d",
            concurrency_id="gpu",
            concurrency_limit=1,
        )
    return demo
