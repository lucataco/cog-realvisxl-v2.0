from cog import BasePredictor, Input, Path
import os
import torch
import time
from diffusers import (DDIMScheduler, 
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler,
    HeunDiscreteScheduler, 
    PNDMScheduler)

MODEL_NAME = "SG161222/RealVisXL_V2.0"
MODEL_CACHE = "model-cache"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.txt2img_pipe.to("cuda")
        t2 = time.time()
        print("Setup sdxl took: ", t2 - t1)


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="DPMSolverMultistep",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=40
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=10, default=7
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        sdxl_kwargs = {}
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        pipe = self.txt2img_pipe

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }
        output = pipe(**common_args, **sdxl_kwargs)

        output_path = f"/tmp/output.png"
        output.images[0].save(output_path)

        return Path(output_path)