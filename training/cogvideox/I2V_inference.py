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
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler

from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image, load_video

from args import get_args 
from diffusers.pipelines.cogvideo import CogVideoXImageToVideoPipeline
from diffusers.models.transformers import CogVideoXTransformer3DModel
from schedulers.trajectory_scheduler import CogVideoXControlnetDPMScheduler
from pipelines.inverse_pipeline import InversePipeline
from utils import save_tensor_as_video, save_tensor_as_images_with_pca, load_frames_as_tensor
from freeinit_utils import get_freq_filter, freq_mix_3d
from einops import rearrange

@torch.no_grad()
def latent_shift(
    z,  # [B C T H W]
    frame0_noise,  # [B C H W]
    bounding_boxes,  # List of bounding boxes
    first_frame_boxes # List of first frame bounding boxes
):
    assert len(bounding_boxes) == z.shape[0], "batch number of bounding boxes should be equal to batch size"
    B, C, T, H, W = z.shape

    for b, (batch_boxes, batch_first_frame_boxes) in enumerate(zip(bounding_boxes, first_frame_boxes)):
        count = torch.zeros((T, H, W), device=z.device)  # 计数器
        for t, frame_boxes in enumerate(batch_boxes):
            print(len(frame_boxes))
            if t == 0:
                continue
            for box, first_box in zip(frame_boxes, batch_first_frame_boxes):
                x1, y1, x2, y2 = box
                fx1, fy1, fx2, fy2 = first_box
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
                fx1, fy1, fx2, fy2 = int(fx1 * W), int(fy1 * H), int(fx2 * W), int(fy2 * H)
                # 提取 frame0_noise 的对应部分并缩放到目标大小
                noise_patch = frame0_noise[b, :, fy1:fy2, fx1:fx2].unsqueeze(0)  # [1, C, H, W]
                scaled_noise_patch = F.interpolate(noise_patch, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False).squeeze(0)
                z[b, :, t, y1:y2, x1:x2] += scaled_noise_patch
                count[t, y1:y2, x1:x2] += 1

        # 对重叠区域进行平均处理
        for t in range(1, T):
            mask = count[t] > 0
            z[b, :, t, mask] /= count[t, mask]
    return z

@torch.no_grad()
def init_filter(video_length, height, width, channels, filter_params, device):
    # initialize frequency filter for noise reinitialization
    batch_size = 1
    num_channels_latents = channels
    filter_shape = [
        batch_size, 
        num_channels_latents, 
        video_length - 1, 
        height, 
        width
    ]
    # self.freq_filter = get_freq_filter(filter_shape, device=self._execution_device, params=filter_params)
    freq_filter = get_freq_filter(
        filter_shape, 
        device=device, 
        filter_type=filter_params["method"],
        n=filter_params["n"] if filter_params["method"]=="butterworth" else None,
        d_s=filter_params["d_s"],
        d_t=filter_params["d_t"]
    )
    return freq_filter

def main(args):
    model_card = "THUDM/CogVideoX-5b-I2V"
    
    tokenizer    = T5Tokenizer.from_pretrained(model_card, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(model_card, subfolder="text_encoder").cuda()
    vae          = AutoencoderKLCogVideoX.from_pretrained(model_card, subfolder="vae").cuda()
    transformer  = CogVideoXTransformer3DModel.from_pretrained(model_card, subfolder="transformer", torch_dtype=torch.bfloat16)
    scheduler    = CogVideoXControlnetDPMScheduler.from_pretrained(model_card, subfolder="scheduler")
    pipe         = InversePipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, scheduler=scheduler).to(torch.bfloat16)

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
        validation_trajectory_maps = args.validation_trajectory_maps.split(args.validation_prompt_separator)
        for validation_image, validation_prompt, validation_trajectory_map in zip(validation_images, validation_prompts, validation_trajectory_maps):
            image = load_image(validation_image)
            src_video = list(image for _ in range(num_frames))
            pipeline_args = {
                "image": image,
                "prompt": validation_prompt,
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
                "num_frames": num_frames,
            }
            # inverse image latents and then do latent shift
            latents = None
            latents = pipe.inverse(**pipeline_args, src_video=src_video, num_inference_steps=5,).frames
            latents[:,1:,:,:,:] = torch.randn_like(latents[:,1:,:,:,:])
            latents = rearrange(latents, "B T C H W -> B C T H W")

            bounding_boxes = []
            first_frame_boxes = []
            bounding_box, first_frame_bounding_box = load_frames_as_tensor("/home/qid/quanhao/workspace/Open-Sora/assets/boxs_trajectory/boat/moved_mask_boxes", num_frames)
            bounding_boxes.append(bounding_box)
            first_frame_boxes.append(first_frame_bounding_box) 

            frame0_noise = latents[:,:,0,:,:]
            z_shift = latent_shift(latents, frame0_noise, bounding_boxes, first_frame_boxes)
            z_rand = torch.randn_like(latents[:,:,1:,:,:])
            filter_params = {
                "method": "butterworth",
                "n": 4,
                "d_s": 0.75,
                "d_t": 0.75
            }
            z_FFT = freq_mix_3d(z_shift[:,:,1:,:,:].to(dtype=torch.float32), z_rand.to(dtype=torch.float32), LPF=init_filter(13, 60, 90, 16, filter_params, device=pipe.device))
            latents[:,:,1:,:,:] = z_FFT
            latents = rearrange(latents, "B C T H W -> B T C H W")
            output_path = "samples/inverse/noise"
            save_tensor_as_images_with_pca(latents, output_path)
            
            video_generate = pipe(
                **pipeline_args,
                latents=latents,
                generator=torch.Generator(device=pipe.device).manual_seed(args.seed),  # Set the seed for reproducibility
                output_type="np",
            ).frames[0]
            output_path = "samples/inverse/boat_inverse.mp4"
            export_to_video(video_generate, output_path, fps=fps)

if __name__ == "__main__":
    args = get_args()
    main(args)
