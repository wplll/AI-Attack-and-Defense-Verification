import sys
import requests
import io
from PIL import Image
from cfg import flux_path
from diffusers import FluxPipeline


def generate_image(prompt):
    pipe = FluxPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    image_path = "generated_flux_image.png"
    image.save(image_path)
    
    return image_path

if __name__ == "__main__":
    
    prompt = sys.argv[1]
    image_path = generate_image(prompt)
    print(image_path)
