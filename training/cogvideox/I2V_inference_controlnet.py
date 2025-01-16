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

from diffusers.utils import export_to_video, load_image, load_video

from args import get_args 
from pipelines.pipeline_controlnet import CogVideoXImageToVideoControlnetPipeline
from models.transformer_controlnet import CogVideoXControlnetTransformer3DModel
from models.controlnet import CogVideoXControlnet

def main(args):
    model_card = "THUDM/CogVideoX-5b-I2V"
    
    tokenizer    = T5Tokenizer.from_pretrained(model_card, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_card, subfolder="text_encoder").cuda()
    vae          = AutoencoderKLCogVideoX.from_pretrained(model_card, subfolder="vae").cuda()
    transformer  = CogVideoXControlnetTransformer3DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16)
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    controlnet   = CogVideoXControlnet(**model_config,)
    if args.init_from_transformer:
        controlnet_state_dict = {}
        for name, params in transformer.state_dict().items():
            controlnet_state_dict[name] = params
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from transformer was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')

    elif args.pretrained_controlnet_path:
        ckpt = torch.load(args.pretrained_controlnet_path, map_location='cpu', weights_only=False)
        controlnet_state_dict = {}
        for name, params in ckpt['state_dict'].items():
            controlnet_state_dict[name] = params
        m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
        print(f'[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]')
    params = [p.numel() for n, p in controlnet.named_parameters()]
    print(f"### Whole Controlnet Parameters: {sum(params) / 1e9} B")
    params = [p.numel() for n, p in transformer.named_parameters()]
    print(f"### Whole Transformer Parameters: {sum(params) / 1e9} B")

    scheduler    = CogVideoXDPMScheduler.from_pretrained(model_card, subfolder="scheduler")
    pipe         = CogVideoXImageToVideoControlnetPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, controlnet=controlnet, scheduler=scheduler).to(torch.bfloat16)


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
    if args.validation_args_csv and args.num_validation_videos > 0:
        import pandas as pd
        validation_args = pd.read_csv(args.validation_args_csv)
        validation_prompts = validation_args['validation_prompt'].tolist()
        validation_images = validation_args['validation_images'].tolist()
        validation_trajectory_maps = validation_args['validation_trajectory_maps'].tolist()
        output_paths = validation_args['output_path'].tolist()
        controlnet_weights = validation_args['controlnet_weights'].tolist()
        for validation_image, validation_prompt, validation_trajectory_map, output_path, weight in zip(validation_images, validation_prompts, validation_trajectory_maps, output_paths, controlnet_weights):
            pipeline_args = {
                "image": load_image(validation_image),
                "prompt": validation_prompt,
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
                "max_sequence_length": model_config.max_text_seq_length,
                "trajectory_maps": load_video(validation_trajectory_map),
                "controlnet_weights": float(weight),
            }

            video_generate = pipe(
                **pipeline_args,
                num_frames=num_frames,
                generator=torch.Generator(device=pipe.device).manual_seed(args.seed),  # Set the seed for reproducibility
                output_type="np",
            ).frames[0]

            export_to_video(video_generate, output_path, fps=fps)

if __name__ == "__main__":
    args = get_args()
    main(args)
