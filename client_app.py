#!/usr/bin/env python
import gradio as gr
from gradio_client import Client, handle_file

DESCRIPTION = """
![Nosana Banner](https://gcdnb.pbrd.co/images/0juVIv58VlRy.png) 
# Nosana 3D Generation Demo    
"""


def run_text_to_3d(prompt, seed, guidance_scale, num_inference_steps):
    client = Client("hysts/Shap-E")
    result = client.predict(
        prompt=prompt,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        api_name="/text-to-3d",
    )
    return result


def run_image_to_3d(image_url, seed, guidance_scale, num_inference_steps):
    client = Client("hysts/Shap-E")
    result = client.predict(
        image=handle_file(image_url),
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        api_name="/image-to-3d",
    )
    return result


with gr.Blocks(css_paths="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.Tab("Text to 3D"):
            with gr.Group():
                with gr.Row(elem_id="prompt-container"):
                    inp_prompt = gr.Text(label="Prompt", placeholder="Enter text prompt", value="an elephant")
                out_text = gr.Model3D(label="Output", show_label=False)
                with gr.Accordion("Advanced options", open=False):
                    inp_seed = gr.Slider(label="Seed", minimum=0, maximum=100, step=1, value=0)
                    inp_guidance = gr.Slider(label="Guidance scale", minimum=1, maximum=30, step=1, value=15)
                    inp_steps = gr.Slider(label="Inference steps", minimum=10, maximum=100, step=1, value=64)
                run_text_btn = gr.Button("Run Text to 3D")
            run_text_btn.click(
                run_text_to_3d, inputs=[inp_prompt, inp_seed, inp_guidance, inp_steps], outputs=out_text
            )
        with gr.Tab("Image to 3D"):
            with gr.Group():

                inp_image = gr.Image(label="Input image", type="filepath")
                out_image = gr.Model3D(label="Output", show_label=False)
                with gr.Accordion("Advanced options", open=False):
                    inp_seed_img = gr.Slider(label="Seed", minimum=0, maximum=100, step=1, value=0)
                    inp_guidance_img = gr.Slider(label="Guidance scale", minimum=1, maximum=30, step=0.1, value=3)
                    inp_steps_img = gr.Slider(label="Inference steps", minimum=10, maximum=100, step=1, value=64)
                run_img_btn = gr.Button("Run Image to 3D")
            run_img_btn.click(
                run_image_to_3d, inputs=[inp_image, inp_seed_img, inp_guidance_img, inp_steps_img], outputs=out_image
            )

if __name__ == "__main__":
    demo.launch()
