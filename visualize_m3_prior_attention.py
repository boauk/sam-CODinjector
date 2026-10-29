# -*- coding: utf-8 -*-
"""Inspect where trained M3 image tokens attend within DINO prior tokens.

This script does not change training code or checkpoint parameters. Attention
weights are captured during inference and reduced immediately to spatial maps.
"""

import argparse
import gc
import json
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
from vlm_prior import (
    GroundingDinoRuntimePriorGenerator,
    LazyGroundingDinoRuntimePriorGenerator,
    PriorCache,
    load_or_generate_prior,
    resolve_prior_cache_dir,
)

M3_DIR = Path(__file__).resolve().parent / "消融" / "M3"
sys.path.insert(0, str(M3_DIR))
from vlsam_dino_model import VLSAMDino  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize trained M3 cross-attention to DINO box tokens"
    )
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
    parser.add_argument("-model_type", "--model-type", type=str, default="vit_h", choices=("vit_b", "vit_l", "vit_h"))
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
    parser.add_argument("--image", required=True)
    parser.add_argument("--gt", default="", help="Optional ground-truth mask")
    parser.add_argument("--output", default="outputs/m3_prior_attention.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--save-heads", action="store_true", help="Also save each attention head")
    parser.add_argument("--prior-cache-dir", default=None)
    parser.add_argument("--manual-description", default="")
    parser.add_argument(
        "--blip-model-name",
        default=GroundingDinoRuntimePriorGenerator.DEFAULT_BLIP_MODEL,
    )
    parser.add_argument(
        "--dino-model-name",
        default=GroundingDinoRuntimePriorGenerator.DEFAULT_DINO_MODEL,
    )
    return parser.parse_args()


def resolve_trained_checkpoint(args):
    """Resolve the trained M3 checkpoint from the explicit path or work_dir."""
    if not args.load_pretrain:
        return None
    if args.pretrain_model_path:
        return Path(args.pretrain_model_path)
    return Path(args.work_dir) / args.task_name / "vlsam_model_best.pth"


def image_input(path):
    image = Image.open(path).convert("RGB")
    tensor = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )(image).unsqueeze(0)
    return np.asarray(image), tensor


def release_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def amp_context(args, device):
    if args.use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def prior_occupancy(hard_prior, grid_size):
    """Approximate each context token's box coverage in the original image."""
    if hard_prior.ndim == 3:
        hard_prior = hard_prior.unsqueeze(0)
    return F.adaptive_avg_pool2d(hard_prior.float(), grid_size)[0, 0].flatten()


def summarize_weights(weights, occupancy, query_shape):
    """Reduce [B, heads, queries, keys] before the next encoder block runs."""
    if weights.ndim != 4 or weights.shape[0] != 1:
        raise ValueError("Expected attention weights [1, heads, queries, keys]")
    height, width = query_shape
    if weights.shape[2] != height * width or weights.shape[3] != occupancy.numel():
        raise ValueError("Attention token counts do not match the query/prior grids")

    box_coverage = occupancy.mean().item()
    if box_coverage <= 0:
        raise ValueError("DINO box does not cover any prior tokens")
    inside = torch.matmul(
        weights,
        occupancy.to(device=weights.device, dtype=weights.dtype),
    )[0].float().reshape(weights.shape[1], height, width)
    per_head = inside.detach().cpu().numpy()
    query_map = per_head.mean(axis=0)
    return {
        "query_inside_mass": query_map,
        "head_inside_mass": per_head,
        "mean_inside_mass": float(query_map.mean()),
        "prior_token_coverage": float(box_coverage),
        "enrichment": float(query_map.mean() / box_coverage),
    }


def capture_encoder_attention(model, image, hard_prior, save_heads=False):
    encoder = model.image_encoder
    indexes = [int(i) for i in encoder.prior_injection_indexes]
    soft_prior = model._soften_prior_mask(hard_prior)
    prior_context = model.prior_context_encoder(soft_prior)
    occupancy = prior_occupancy(hard_prior, prior_context.shape[-2:])

    patch_height, patch_width = encoder.patch_embed.proj.stride
    query_shape = (image.shape[-2] // patch_height, image.shape[-1] // patch_width)
    attention = encoder.prior_injector.attn
    original_forward = attention.forward
    had_instance_forward = "forward" in vars(attention)
    results = {}
    call_index = 0

    def capture_forward(query, key, value, **kwargs):
        nonlocal call_index
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        output, weights = original_forward(query, key, value, **kwargs)
        if weights is None or call_index >= len(indexes):
            raise RuntimeError("Could not match captured attention to an injection block")
        result = summarize_weights(weights, occupancy, query_shape)
        if not save_heads:
            result.pop("head_inside_mass")
        results[indexes[call_index]] = result
        call_index += 1
        return output, weights

    try:
        attention.forward = capture_forward
        encoder(image, prior_context=prior_context)
    finally:
        if had_instance_forward:
            attention.forward = original_forward
        else:
            del attention.forward

    if call_index != len(indexes):
        raise RuntimeError(f"Captured {call_index} of {len(indexes)} injection blocks")
    return results, hard_prior


def resize_map(array, size):
    array = np.asarray(array, dtype=np.float32)
    image = Image.fromarray(array, mode="F")
    return np.asarray(image.resize(size, Image.BILINEAR), dtype=np.float32)


def plot_results(output_path, rgb, hard_prior, results, gt=None, save_heads=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    height, width = rgb.shape[:2]
    mask = resize_map(hard_prior[0, 0].cpu().numpy(), (width, height))
    rgb_float = rgb.astype(np.float32) / 255.0
    box_image = rgb_float.copy()
    box_edge = np.logical_xor(mask > 0.5, np.roll(mask > 0.5, 1, axis=0)) | np.logical_xor(
        mask > 0.5, np.roll(mask > 0.5, 1, axis=1)
    )
    box_image[box_edge] = (1.0, 0.12, 0.12)

    indexes = sorted(results)
    maps = {index: results[index]["query_inside_mass"] for index in indexes}
    coverage = results[indexes[0]]["prior_token_coverage"]
    enrichment_maps = {index: values / coverage for index, values in maps.items()}
    upper = max(1.0, max(float(np.max(values)) for values in enrichment_maps.values()))
    panel_count = len(indexes) + 1 + int(gt is not None)
    fig, axes = plt.subplots(1, panel_count, figsize=(3.5 * panel_count, 4.4), constrained_layout=True)
    axes[0].imshow(box_image)
    axes[0].set_title("Image + DINO box")
    axes[0].axis("off")
    plt.imsave(parts_dir / "image_box.png", box_image)

    offset = 1
    if gt is not None:
        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        offset += 1
        plt.imsave(parts_dir / "ground_truth.png", gt, cmap="gray", vmin=0, vmax=1)

    metadata = {
        "definition": "mean attention mass to DINO-box prior tokens; coverage from pooled hard box",
        "warning": "Context tokens are convolutional features without explicit 2D positional encoding; box coverage is approximate.",
        "box_vs_background_comparable": coverage < 1.0,
        "blocks": {},
    }
    if coverage >= 1.0:
        print("Warning: DINO prior covers the whole image; inside/outside comparison is undefined.")
    for column, index in enumerate(indexes, start=offset):
        result = results[index]
        normalized = resize_map(enrichment_maps[index], (width, height))
        colored = plt.get_cmap("turbo")(np.clip(normalized / upper, 0, 1))[..., :3]
        overlay = 0.45 * rgb_float + 0.55 * colored
        axes[column].imshow(overlay)
        axes[column].set_title(
            f"Block {index}\ninside {result['mean_inside_mass']:.3f} / area {coverage:.3f}"
        )
        axes[column].axis("off")
        plt.imsave(parts_dir / f"block_{index:02d}_overlay.png", overlay)
        np.save(parts_dir / f"block_{index:02d}_query_inside_mass.npy", maps[index])
        if save_heads:
            for head, head_map in enumerate(result["head_inside_mass"]):
                plt.imsave(
                    parts_dir / f"block_{index:02d}_head_{head:02d}.png",
                    head_map / coverage,
                    cmap="turbo",
                    vmin=0,
                    vmax=upper,
                )
        metadata["blocks"][str(index)] = {
            key: value
            for key, value in result.items()
            if key not in ("query_inside_mass", "head_inside_mass")
        }

    scalar = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=0, vmax=upper))
    fig.colorbar(scalar, ax=axes[offset:].tolist(), fraction=0.025, pad=0.015, label="Box-token attention / area baseline")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved attention figure: {output_path.resolve()}")
    print(f"Saved statistics: {metadata_path.resolve()}")
    print(f"Saved per-block panels: {parts_dir.resolve()}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    image_path = Path(args.image)
    rgb, image_cpu = image_input(image_path)

    cache_dir = resolve_prior_cache_dir(args.prior_cache_dir, image_path.parent)
    prior_generator = LazyGroundingDinoRuntimePriorGenerator(
        blip_model_name=args.blip_model_name,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    prior_cache = PriorCache(cache_dir) if cache_dir else None
    prior = load_or_generate_prior(
        prior_generator,
        Image.fromarray(rgb),
        image_path,
        target_text=args.manual_description or None,
        prior_cache=prior_cache,
    )
    hard_prior_cpu = prior.prior_mask.detach().cpu()
    del prior, prior_generator
    release_cache()

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    model = VLSAMDino(
        image_encoder=sam.image_encoder,
        mask_decoder=sam.mask_decoder,
        text_feature_dim=256,
    )
    trained_checkpoint = resolve_trained_checkpoint(args)
    if trained_checkpoint is None or not trained_checkpoint.is_file():
        raise FileNotFoundError(
            "Trained M3 checkpoint not found. Set -pretrain_model_path, or place "
            f"vlsam_model_best.pth at {trained_checkpoint}."
        )
    checkpoint = torch.load(trained_checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    model = model.to(device).eval()

    with torch.inference_mode(), amp_context(args, device):
        hard_prior = hard_prior_cpu.to(device=device, dtype=image_cpu.dtype)
        if hard_prior.ndim == 3:
            hard_prior = hard_prior.unsqueeze(0)
        results, _ = capture_encoder_attention(
            model, image_cpu.to(device), hard_prior, save_heads=args.save_heads
        )

    gt = None
    if args.gt:
        with Image.open(args.gt) as gt_image:
            gt = np.asarray(
                gt_image.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST),
                dtype=np.float32,
            ) / 255.0
    plot_results(args.output, rgb, hard_prior_cpu, results, gt=gt, save_heads=args.save_heads)


if __name__ == "__main__":
    main()
