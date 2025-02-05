# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
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

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import PIL
import PIL.Image
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline, retrieve_timesteps, retrieve_latents
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from utils import save_tensor_as_images_with_pca

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class InverseI2VPipeline(CogVideoXImageToVideoPipeline):

    @torch.no_grad()
    def encode_video(
        self,
        src_video: torch.Tensor,
        height: int = 480,
        width: int = 720,
        batch_size: int = 1,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        generator: Optional[torch.Generator] = None,
    ):
        device = self._execution_device
        src_video = self.video_processor.preprocess_video(src_video, height=height, width=width).to(
            device, dtype=dtype
        )

        if isinstance(generator, list):
            video_latents = [
                retrieve_latents(self.vae.encode(src_video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            video_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator) for img in src_video]

        video_latents = torch.cat(video_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]

        if not self.vae.config.invert_scale_latents:
            video_latents = self.vae_scaling_factor_image * video_latents
        else:
            # This is awkward but required because the CogVideoX team forgot to multiply the
            # scaling factor during training :)
            video_latents = 1 / self.vae_scaling_factor_image * video_latents

        # Select the first frame along the second dimension
        if self.transformer.config.patch_size_t is not None:
            first_frame = video_latents[:, : video_latents.size(1) % self.transformer.config.patch_size_t, ...]
            video_latents = torch.cat([first_frame, video_latents], dim=1)

        return video_latents