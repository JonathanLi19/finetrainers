# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler

from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image

from args import get_args 
from pipelines.pipeline_trajectory import CogVideoXTrajectoryImageToVideoPipeline
from models.transformer_trajectory import CogVideoXTrajectoryTransformer3DModel
from numerize import numerize
from diffusers import (
    CogVideoXTransformer3DModel,
)
import time

def main(args):
    model_card = "THUDM/CogVideoX-5b-I2V"
    output_path = "output/inference/rocket_landing_v1.mp4"
    
    tokenizer    = T5Tokenizer.from_pretrained(model_card, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_card, subfolder="text_encoder").cuda()
    vae          = AutoencoderKLCogVideoX.from_pretrained(model_card, subfolder="vae").cuda()
    transformer  = CogVideoXTrajectoryTransformer3DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler    = CogVideoXDPMScheduler.from_pretrained(model_card, subfolder="scheduler")
    pipe         = CogVideoXTrajectoryImageToVideoPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, scheduler=scheduler).to(torch.bfloat16)

    if model_card == "THUDM/CogVideoX1.5-5B-I2V":
        num_frames = 81
        fps = 16
    elif model_card == "THUDM/CogVideoX-5b-I2V":
        num_frames = 49
        fps = 8

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    pipe.to("cuda")
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Run inference
    if args.validation_prompt and args.num_validation_videos > 0:
        validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
        validation_images = args.validation_images.split(args.validation_prompt_separator)
        for validation_image, validation_prompt in zip(validation_images, validation_prompts):
            pipeline_args = {
                "image": load_image(validation_image),
                "prompt": validation_prompt,
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
            }

            video_generate = pipe(
                **pipeline_args,
                num_videos_per_prompt=1,  # Number of videos to generate per prompt
                num_inference_steps=50,  # Number of inference steps
                num_frames=num_frames,  # Number of frames to generate
                generator=torch.Generator().manual_seed(args.seed),  # Set the seed for reproducibility
                trajectory_maps=torch.zeros(num_frames, 3, args.height, args.width),  # Set the trajectory map to zeros
                trajectory_guidance_scale=2,
            ).frames[0]

            export_to_video(video_generate, output_path, fps=fps)

if __name__ == "__main__":
    args = get_args()
    main(args)
