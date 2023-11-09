# SG161222/RealVisXL_V2.0 Cog model

This is an implementation of the [SG161222/RealVisXL_V2.0](https://huggingface.co/SG161222/RealVisXL_V2.0) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i prompt="dark shot, front shot, closeup photo of a 25 y.o latino man, perfect eyes, natural skin, skin moles, looks at viewer, cinematic shot" -i seed=1111316861

With a RealvisXL-v2 LoRA:

    cog predict -i prompt="photo of a TOK" -i lora_weights="https://replicate.delivery/pbxt/L5zHkM0OHX4ZF1Ipnaiok6GHGvrRgZHBqbz2JjtBAtWz8mdE/trained_model.tar"

## Example:

"dark shot, front shot, closeup photo of a 25 y.o latino man, perfect eyes, natural skin, skin moles, looks at viewer, cinematic shot"

![alt text](output.png)
