import sys
import torch
from diffusers import KolorsPipeline
from PIL import Image
from cfg import kolors_path

def generate_image(prompt):
    pipe = KolorsPipeline.from_pretrained(
        kolors_path, 
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to("cuda")
    
    image = pipe(
        prompt=prompt,
        negative_prompt="",
        guidance_scale=5.0,
        num_inference_steps=50,
        generator=torch.Generator(pipe.device).manual_seed(66),
    ).images[0]
    
    image_path = "generated_kolors_image.png"
    image.save(image_path)
    
    return image_path

if __name__ == "__main__":
    prompt = sys.argv[1]

    image_path = generate_image(prompt)
    print(image_path)
