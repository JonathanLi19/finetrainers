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

import gc
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel


from args import get_args  # isort:skip
from dataset import BucketSampler, VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
)


def log_validation(
    pipe: CogVideoXImageToVideoPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    is_final_validation: bool = False,
):

    pipe = pipe.to("cuda")

    # run inference
    generator = torch.Generator(device="cuda").manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    phase_name = "test" if is_final_validation else "validation"
    video_filenames = []
    for i, video in enumerate(videos):
        prompt = (
            pipeline_args["prompt"][:25]
            .replace(" ", "_")
            .replace(" ", "_")
            .replace("'", "_")
            .replace('"', "_")
            .replace("/", "_")
        )
        filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
        export_to_video(video, filename, fps=8)
        video_filenames.append(filename)

def main(args):
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights(args.pretrained_model_name_or_path, adapter_name="cogvideox-lora")
    pipe.set_adapters(["cogvideox-lora"], [1.0])
    del pipe.transformer.patch_embed.pos_embedding
    pipe.transformer.patch_embed.use_learned_positional_embeddings = False
    pipe.transformer.config.use_learned_positional_embeddings = False

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

            log_validation(
                pipe=pipe,
                args=args,
                pipeline_args=pipeline_args,
            )

if __name__ == "__main__":
    args = get_args()
    main(args)
