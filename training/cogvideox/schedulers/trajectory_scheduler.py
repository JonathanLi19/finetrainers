import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

class CogVideoXControlnetDPMScheduler(CogVideoXDPMScheduler):
    def inverse(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[DDIMSchedulerOutput, Tuple]:

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 1. get next step value (=t+1)
        next_timestep = timestep + self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_next = self.alphas_cumprod[next_timestep]
        if alpha_prod_t_next == 0:
            alpha_prod_t_next = torch.tensor(1e-8, dtype=alpha_prod_t_next.dtype)
        alpha_prod_t_back = None
        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        # To make style tests pass, commented out `pred_epsilon` as it is an unused variable
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            # pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            # pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        h, r, lamb, lamb_next = self.get_variables(alpha_prod_t_next, alpha_prod_t, alpha_prod_t_back)
        mult = list(self.get_mult(h, r, alpha_prod_t_next, alpha_prod_t, alpha_prod_t_back))
        mult_noise = (1 - alpha_prod_t) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5

        noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
        prev_sample = sample / mult[0] + (mult[1] / mult[0]) * pred_original_sample - (mult_noise / mult[0]) * noise

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)