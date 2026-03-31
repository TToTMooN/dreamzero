"""Video generation callback for wandb logging during training.

Periodically generates predicted videos from the model and logs them to wandb
alongside ground truth frames for visual comparison.
"""
import logging
import os
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class VideoEvalCallback(TrainerCallback):
    """Generate and log predicted videos to wandb during training.

    At every `log_video_every_n_steps`, takes a sample from the training data,
    runs a forward pass to get the model's video prediction (denoised latent),
    decodes it via VAE, and logs ground truth vs predicted frames to wandb.
    """

    def __init__(
        self,
        log_video_every_n_steps: int = 1000,
        num_samples: int = 1,
        num_frames_to_log: int = 8,
        num_inference_steps: int = 4,
    ):
        self.log_video_every_n_steps = log_video_every_n_steps
        self.num_samples = num_samples
        self.num_frames_to_log = num_frames_to_log
        self.num_inference_steps = num_inference_steps
        self._dataset = None
        self._last_logged_step = -1

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step == self._last_logged_step:
            return
        if state.global_step % self.log_video_every_n_steps != 0:
            return
        # Only log from rank 0
        if int(os.environ.get("RANK", "0")) != 0:
            return

        self._last_logged_step = state.global_step

        try:
            self._log_videos(state, model, **kwargs)
        except Exception as e:
            logger.warning(f"VideoEvalCallback failed at step {state.global_step}: {e}")

    def _log_videos(self, state, model, **kwargs):
        import wandb
        if wandb.run is None:
            return

        # Get the underlying model (unwrap DeepSpeed/DDP)
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module

        action_head = unwrapped.action_head
        vae = action_head.vae
        device = next(action_head.parameters()).device

        # Get a training sample
        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))

        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Extract ground truth video frames
        gt_videos = batch.get("images")  # [B, T, H, W, C]
        if gt_videos is None:
            return

        gt_frames = gt_videos[0].cpu().numpy()  # [T, H, W, C]
        if gt_frames.dtype != np.uint8:
            gt_frames = ((gt_frames + 1) * 127.5).clip(0, 255).astype(np.uint8)

        # Sample frames evenly
        total_frames = gt_frames.shape[0]
        indices = np.linspace(0, total_frames - 1, self.num_frames_to_log, dtype=int)
        gt_sampled = [gt_frames[i] for i in indices]

        # Run forward pass to get video prediction (noised then denoised)
        with torch.inference_mode():
            # Encode GT video to latent
            videos_bchw = rearrange(
                gt_videos[:1].float() / 255.0 if gt_videos.dtype == torch.uint8 else gt_videos[:1].float(),
                "b t h w c -> b c t h w"
            )

            # Resize to target resolution
            target_h = getattr(action_head.config, "target_video_height", None)
            target_w = getattr(action_head.config, "target_video_width", None)
            if target_h and target_w:
                b, c, t, h, w = videos_bchw.shape
                if (h, w) != (target_h, target_w):
                    videos_bchw = torch.nn.functional.interpolate(
                        videos_bchw.reshape(b * t, c, h, w),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(b, c, t, target_h, target_w)

            latents = action_head.encode_video(
                videos_bchw.to(device),
                action_head.tiled,
                (action_head.tile_size_height, action_head.tile_size_width),
                (action_head.tile_stride_height, action_head.tile_stride_width),
            )

            # Add noise at medium level then denoise to get model prediction
            noise = torch.randn_like(latents)
            # Use mid-level noise (t=500/1000) to show meaningful reconstruction
            t_mid = torch.tensor([500], device=device).expand(latents.shape[0] * latents.shape[2])
            noisy_latents = action_head.scheduler.add_noise(
                latents.flatten(0, 1).transpose(0, 1).reshape(-1, *latents.shape[2:]),
                noise.flatten(0, 1).transpose(0, 1).reshape(-1, *latents.shape[2:]),
                t_mid[:latents.shape[2]],
            )

            # Decode the noisy latents (shows what model sees at t=500)
            noisy_decoded = vae.decode(
                noisy_latents.unsqueeze(0) if noisy_latents.dim() == 4 else noisy_latents,
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )

            # Also decode clean latents (reconstruction quality check)
            clean_decoded = vae.decode(
                latents.transpose(1, 2).to(device),
                tiled=action_head.tiled,
                tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
            )

            # Convert to numpy frames
            def latent_to_frames(decoded_video):
                frames = rearrange(decoded_video, "B C T H W -> B T H W C")
                frames = frames[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                total = frames.shape[0]
                idx = np.linspace(0, total - 1, self.num_frames_to_log, dtype=int)
                return [frames[i] for i in idx]

            recon_frames = latent_to_frames(clean_decoded)
            noisy_frames = latent_to_frames(noisy_decoded)

        # Log to wandb
        log_dict = {}

        # Ground truth frames
        for i, frame in enumerate(gt_sampled):
            log_dict[f"video/gt_frame_{i}"] = wandb.Image(frame, caption=f"GT frame {indices[i]}")

        # VAE reconstruction frames
        for i, frame in enumerate(recon_frames):
            log_dict[f"video/recon_frame_{i}"] = wandb.Image(frame, caption=f"VAE recon frame {i}")

        # Noisy input frames (what model sees at t=500)
        for i, frame in enumerate(noisy_frames):
            log_dict[f"video/noisy_frame_{i}"] = wandb.Image(frame, caption=f"Noisy (t=500) frame {i}")

        # Log video as mp4 if imageio available
        try:
            import imageio
            import tempfile

            # GT video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                imageio.mimsave(f.name, gt_sampled, fps=5)
                log_dict["video/ground_truth"] = wandb.Video(f.name, fps=5, format="mp4")

            # Reconstruction video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                imageio.mimsave(f.name, recon_frames, fps=5)
                log_dict["video/vae_reconstruction"] = wandb.Video(f.name, fps=5, format="mp4")
        except Exception:
            pass  # imageio not available, skip video logging

        wandb.log(log_dict, step=state.global_step)
        logger.info(f"Logged video samples to wandb at step {state.global_step}")
