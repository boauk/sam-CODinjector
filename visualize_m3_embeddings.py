# -*- coding: utf-8 -*-
"""Visualize SAM/M3 embeddings for a trained checkpoint.

This script is for post-training analysis. It does not update model weights.
"""

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from segment_anything import sam_model_registry

# M3 keeps its model wrapper in the ablation directory, while SAM itself is
# imported from the project package.
M3_DIR = Path(__file__).resolve().parent / "消融" / "M3"
sys.path.insert(0, str(M3_DIR))
from vlsam_dino_model import VLSAMDino  # noqa: E402

from vlm_prior import (  # noqa: E402
    GroundingDinoRuntimePriorGenerator,
    LazyGroundingDinoRuntimePriorGenerator,
    PriorCache,
    image_to_pil,
    load_or_generate_prior,
    resolve_prior_cache_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize M3 feature heatmaps")
    parser.add_argument(
        "-i",
        "--tr_npy_path",
        type=str,
        default="data/CAMO_COD10K",
        help="dataset root directory",
    )
    parser.add_argument("--train_img_dir", type=str, default="Train/Image/")
    parser.add_argument("--train_gt_dir", type=str, default="Train/GT/")
    parser.add_argument("--test_img_dir", type=str, default="Test/Image/")
    parser.add_argument("--test_gt_dir", type=str, default="Test/GT/")
    parser.add_argument("-task_name", "--task_name", type=str, default="camo")
    parser.add_argument("-model_type", "--model-type", type=str, default="vit_h", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument(
        "-checkpoint",
        "--checkpoint",
        type=str,
        default="work_dir/SAM/sam_vit_h_4b8939.pth",
        help="official SAM checkpoint",
    )
    parser.add_argument(
        "--load_pretrain",
        type=bool,
        default=True,
        help="load the trained VLSAM checkpoint",
    )
    parser.add_argument(
        "-pretrain_model_path",
        "--pretrain-model-path",
        type=str,
        default="",
        help="trained M3 checkpoint; falls back to work_dir/task_name/vlsam_model_best.pth",
    )
    parser.add_argument("-work_dir", type=str, default="./work_dir")

    # Keep the image path explicit. Change it in this argument or pass --image
    # on the command line; it is intentionally not hard-coded here.
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--gt", default="", help="Optional ground-truth mask path")
    parser.add_argument("--output", default="outputs/m3_embedding_visualization.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--blip-model-name",
        default=GroundingDinoRuntimePriorGenerator.DEFAULT_BLIP_MODEL,
    )
    parser.add_argument(
        "--dino-model-name",
        default=GroundingDinoRuntimePriorGenerator.DEFAULT_DINO_MODEL,
    )
    parser.add_argument("--prior-cache-dir", default=None)
    parser.add_argument("--manual-description", default="")
    parser.add_argument("--use-amp", action="store_true")
    return parser.parse_args()


def resolve_trained_checkpoint(args):
    """Resolve the trained M3 checkpoint from the explicit path or work_dir."""
    if not args.load_pretrain:
        return None
    if args.pretrain_model_path:
        return Path(args.pretrain_model_path)
    return Path(args.work_dir) / args.task_name / "vlsam_model_best.pth"


def normalize_feature(feature):
    """Convert [B,C,H,W] into a spatial heatmap using channel L2 energy."""
    feature = feature.float()
    heatmap = torch.linalg.vector_norm(feature, dim=1, keepdim=True)
    heatmap = F.interpolate(heatmap, size=(1024, 1024), mode="bilinear", align_corners=False)
    heatmap = heatmap[0, 0].cpu().numpy()
    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-8
    return heatmap


def load_rgb(path):
    return Image.open(path).convert("RGB")


def save_panel(output, image, gt, soft_prior, dense_embedding, image_embedding, prediction):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panels = [(image, "RGB image", None)]
    if gt is not None:
        panels.append((gt, "Ground Truth", "gray"))
    panels.extend(
        [
            (soft_prior, "DINO soft prior", "magma"),
            (dense_embedding, "Dense embedding", "viridis"),
            (image_embedding, "M3 image embedding", "viridis"),
            (prediction, "Predicted mask", "gray"),
        ]
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (panel, title, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(panel)
        else:
            ax.imshow(panel, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    parts_dir = output.parent / f"{output.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(parts_dir / "01_rgb.png")
    if gt is not None:
        plt.imsave(parts_dir / "02_gt.png", gt, cmap="gray", vmin=0, vmax=1)
    plt.imsave(parts_dir / "03_soft_prior.png", soft_prior, cmap="magma", vmin=0, vmax=1)
    plt.imsave(parts_dir / "04_dense_embedding.png", dense_embedding, cmap="viridis", vmin=0, vmax=1)
    plt.imsave(parts_dir / "05_image_embedding.png", image_embedding, cmap="viridis", vmin=0, vmax=1)
    plt.imsave(parts_dir / "06_prediction.png", prediction, cmap="gray", vmin=0, vmax=1)
    print(f"Saved visualization: {output.resolve()}")
    print(f"Saved individual panels: {parts_dir.resolve()}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    image_path = Path(args.image)
    image_pil = load_rgb(image_path)
    image_array = np.asarray(image_pil)
    image_tensor = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(image_pil).unsqueeze(0).to(device)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = VLSAMDino(
        image_encoder=sam.image_encoder,
        mask_decoder=sam.mask_decoder,
        text_feature_dim=256,
    ).to(device)
    trained_checkpoint = resolve_trained_checkpoint(args)
    if trained_checkpoint is None or not trained_checkpoint.is_file():
        raise FileNotFoundError(
            "Trained M3 checkpoint not found. Set -pretrain_model_path, or place "
            f"vlsam_model_best.pth at {trained_checkpoint}."
        )
    checkpoint = torch.load(trained_checkpoint, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    cache_dir = resolve_prior_cache_dir(args.prior_cache_dir, str(image_path.parent))
    prior_generator = LazyGroundingDinoRuntimePriorGenerator(
        blip_model_name=args.blip_model_name,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    prior_cache = PriorCache(cache_dir) if cache_dir else None
    with torch.inference_mode():
        prior = load_or_generate_prior(
            prior_generator,
            image_to_pil(image_array),
            str(image_path),
            target_text=args.manual_description or None,
            prior_cache=prior_cache,
        )
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if args.use_amp and device.type == "cuda"
            else nullcontext()
        )
        with amp_context:
            prior_mask = prior.prior_mask.to(device=device, dtype=image_tensor.dtype)
            if prior_mask.dim() == 3:
                prior_mask = prior_mask.unsqueeze(0)
            soft_prior = model._soften_prior_mask(prior_mask)
            prior_context = model.prior_context_encoder(soft_prior)
            dense_embedding = model.prior_mask_encoder(soft_prior)
            image_embedding = model.image_encoder(image_tensor, prior_context=prior_context)
            text_features = prior.text_features.to(device)
            text_embeddings = model.text_proj(text_features)
            if text_embeddings.dim() == 2:
                text_embeddings = text_embeddings.unsqueeze(0)
            sparse_embeddings = text_embeddings.view(text_embeddings.shape[0], -1, 256)
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.pe_layer(image_embedding.shape[-2:]).unsqueeze(0),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embedding,
                multimask_output=False,
            )
            prediction = torch.sigmoid(
                F.interpolate(
                    low_res_masks,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                )
            )[0, 0].float().cpu().numpy()

    gt = None
    if args.gt:
        gt = np.asarray(Image.open(args.gt).convert("L").resize((1024, 1024), Image.NEAREST), dtype=np.float32) / 255.0
    save_panel(
        args.output,
        image_array,
        gt,
        soft_prior[0, 0].float().cpu().numpy(),
        normalize_feature(dense_embedding),
        normalize_feature(image_embedding),
        prediction,
    )


if __name__ == "__main__":
    main()
