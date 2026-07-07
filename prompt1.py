import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
model_id="runwayml/stable-diffusion-v1-5"
pipe=StableDiffusionPipeline.from_pretrained(
    model_id
).to("cuda")
prompts=[
    "A city above sky with a beautiful gardens",
    "A river can be flow from hill to ground",
    "A person can be jump in hair"
]
for prompt in (prompts):
  result=pipe(prompt).images[0]
  plt.figure(figsize=(7,6))
  plt.imshow(result)
  plt.show()
