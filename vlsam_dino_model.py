from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def structure_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight_kernel_size: int = 31,
    structure_weight: float = 5.0,
) -> torch.Tensor:
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    if targets.shape[-2:] != logits.shape[-2:]:
        targets = F.interpolate(targets, size=logits.shape[-2:], mode="nearest")
    if targets.shape[1] != logits.shape[1]:
        targets = targets.expand_as(logits)

    pad = weight_kernel_size // 2
    local_mean = F.avg_pool2d(
        targets, kernel_size=weight_kernel_size, stride=1, padding=pad
    )
    weights = 1.0 + structure_weight * torch.abs(local_mean - targets)

    wbce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    wbce = (weights * wbce).sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + 1e-8)

    pred = torch.sigmoid(logits)
    inter = ((pred * targets) * weights).sum(dim=(2, 3))
    union = ((pred + targets) * weights).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (wbce + wiou).mean()


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class VLSAMDino(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        text_feature_dim: int = 256,
        prior_text_min_weight: float = 0.1,
        prior_confidence_threshold: float = 0.5,
        prior_confidence_temperature: float = 0.15,
        prior_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.encoder_embed_dim = self._infer_encoder_embed_dim(image_encoder)
        self.pe_layer = PositionEmbeddingRandom(256 // 2)
        self.text_proj = nn.Linear(text_feature_dim, 256)
        self.prior_context_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(128, self.encoder_embed_dim, 3, 2, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        self.prior_mask_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((64, 64)),
        )

    @staticmethod
    def _infer_encoder_embed_dim(image_encoder) -> int:
        patch_embed = getattr(image_encoder, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "proj"):
            proj = patch_embed.proj
            if hasattr(proj, "out_channels"):
                return int(proj.out_channels)
        return 256

    def forward(
        self,
        image: torch.Tensor,
        dino_text_features: torch.Tensor,
        prior_mask: torch.Tensor,
        prior_confidence: Optional[torch.Tensor] = None,
        return_boundary: bool = False,
    ) -> torch.Tensor:
        prior_mask = prior_mask.to(device=image.device, dtype=image.dtype)
        if prior_mask.dim() == 3:
            prior_mask = prior_mask.unsqueeze(0)
        soft_prior = self._soften_prior_mask(prior_mask)
        prior_context = self.prior_context_encoder(soft_prior)
        dense_embeddings = self.prior_mask_encoder(soft_prior)

        image_embedding = self.image_encoder(image, prior_context=prior_context)

        text_embeddings = self.text_proj(dino_text_features)
        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        sparse_embeddings = text_embeddings.view(text_embeddings.shape[0], -1, 256)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.pe_layer(image_embedding.shape[-2:]).unsqueeze(0),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        if return_boundary:
            return masks, None
        return masks

    @staticmethod
    def _soften_prior_mask(prior_mask: torch.Tensor) -> torch.Tensor:
        """Smooth hard box edges while retaining a high response in the box center."""
        prior_mask = prior_mask.clamp(0.0, 1.0)
        kernel_size = 15
        blurred = F.avg_pool2d(
            prior_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        return blurred.clamp(0.0, 1.0)
