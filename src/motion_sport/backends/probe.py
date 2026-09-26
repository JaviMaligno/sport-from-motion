"""Frozen video encoder + tiny probe: the "no training of our own" specialist.

A self-supervised video encoder (V-JEPA 2 by default) is run *frozen* on the same
rendered point-light clips the VLMs see; the only thing fitted is a logistic
regression on top of its pooled embedding, cross-validated by match. If that is
enough, there is no need to design or train a specialist network.

Heavy dependencies (torch, transformers) are optional and imported lazily:

    pip install -e ".[probe]"

Encoders are Hugging Face repos; any model whose processor accepts a list of
frames works. Checked-in defaults:
    facebook/vjepa2-vitl-fpc64-256   V-JEPA 2 ViT-L (uses model.get_vision_features)
    MCG-NJU/videomae-base            VideoMAE (pixel-reconstruction pre-training)
Comparing a motion-predictive encoder (V-JEPA) with a pixel-reconstruction one
(VideoMAE) on point-light input is itself informative.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ENCODER = "facebook/vjepa2-vitl-fpc64-256"


class HFVideoEncoder:
    def __init__(self, repo: str = DEFAULT_ENCODER, num_frames: int = 16, device: str | None = None):
        import torch  # noqa: F401  (fail early with a clear message)
        from transformers import AutoModel, AutoVideoProcessor

        self.repo = repo
        self.num_frames = num_frames
        self.processor = AutoVideoProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo).eval()
        if device:
            self.model.to(device)

    def _resample(self, frames: list[np.ndarray]) -> np.ndarray:
        idx = np.linspace(0, len(frames) - 1, self.num_frames).round().astype(int)
        return np.stack([frames[i] for i in idx])  # [T, H, W, 3] uint8

    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        import torch

        video = self._resample(frames)
        inputs = self.processor(video, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            if hasattr(self.model, "get_vision_features"):  # V-JEPA 2
                feats = self.model.get_vision_features(**inputs)
            else:
                feats = self.model(**inputs).last_hidden_state
        return feats.mean(dim=1).squeeze(0).float().cpu().numpy()


def fit_probe_oof(X: np.ndarray, y: list[str], groups: list[str], n_splits: int = 5,
                  seed: int = 0) -> tuple[list[str], list[dict[str, float]]]:
    """Out-of-fold predictions of a standardised logistic regression, grouped by match."""
    from motion_sport.learners import grouped_oof, logistic_fit_predict

    return grouped_oof(y, groups, logistic_fit_predict(X, y, seed), n_splits)
