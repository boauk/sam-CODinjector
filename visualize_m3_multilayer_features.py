# -*- coding: utf-8 -*-
"""Compare multi-layer feature responses of original SAM and trained M3.

The script reproduces the image-encoder forward order so M3 features are
captured after both the Adapter and prior cross-attention injection.
"""

import argparse
import gc
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from segment_anything import sam_model_registry

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
    parser = argparse.ArgumentParser(
        description="Compare original SAM and M3 features at prior-injection layers"
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
    parser.add_argument("--output", default="outputs/m3_multilayer_features.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-amp", action="store_true")
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
    parser.add_argument(
        "--colormap",
        default="jet",
        help="Matplotlib colormap used for feature heatmaps",
    )
    return parser.parse_args()


def resolve_trained_checkpoint(args):
    """Resolve the trained M3 checkpoint from the explicit path or work_dir."""
    if not args.load_pretrain:
        return None
    if args.pretrain_model_path:
        return Path(args.pretrain_model_path)
    return Path(args.work_dir) / args.task_name / "vlsam_model_best.pth"


def prepare_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image_pil)
    image_tensor = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )(image_pil).unsqueeze(0)
    return image_array, image_tensor


def injection_indexes(encoder):
    indexes = getattr(encoder, "prior_injection_indexes", None)
    if indexes is not None:
        return [int(index) for index in indexes]
    depth = len(encoder.blocks)
    return [min(i * depth // 4, depth - 1) for i in range(4)]


def extract_encoder_stages(encoder, image, indexes, prior_context=None, enhanced=False):
    """Return features after each selected block and optional M3 enhancement."""
    x = encoder.patch_embed(image)
    if encoder.pos_embed is not None:
        x = x + encoder.pos_embed

    selected = {}
    for block_index, block in enumerate(encoder.blocks):
        x = block(x)
        if enhanced:
            group_index = encoder.adapter_group_indexes[block_index]
            adapter_feature = encoder.adapters[group_index](
                x.permute(0, 3, 1, 2)
            ).permute(0, 2, 3, 1)
            adapter_scale = torch.sigmoid(encoder.adapter_gamma[group_index])
            x = x + adapter_scale * adapter_feature

            if prior_context is not None and block_index in indexes:
                prior_feature = encoder.prior_injector(x, prior_context)
                prior_scale = torch.sigmoid(encoder.prior_gamma)
                x = x + prior_scale * prior_feature

        if block_index in indexes:
            selected[block_index] = x.detach().float().cpu()
    return selected


def feature_energy(feature):
    """Collapse [B,H,W,C] to a 2D response map using channel L2 energy."""
    return torch.linalg.vector_norm(feature[0], dim=-1).numpy()


def jointly_normalize(first, second):
    """Use a shared robust range so each SAM/M3 layer pair is comparable."""
    merged = np.concatenate([first.reshape(-1), second.reshape(-1)])
    low, high = np.percentile(merged, [1.0, 99.0])
    if high <= low:
        low, high = float(merged.min()), float(merged.max())
    scale = max(high - low, 1e-8)
    return (
        np.clip((first - low) / scale, 0.0, 1.0),
        np.clip((second - low) / scale, 0.0, 1.0),
    )


def overlay_heatmap(image, heatmap, cmap_name, alpha=0.55):
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        (image.shape[1], image.shape[0]), Image.Resampling.BILINEAR
    )
    heatmap = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    colored = plt.get_cmap(cmap_name)(heatmap)[..., :3]
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip((1.0 - alpha) * rgb + alpha * colored, 0.0, 1.0)


def save_figure(output, image, indexes, sam_maps, m3_maps, cmap_name):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output.parent / f"{output.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        len(indexes) + 1,
        figsize=(3.2 * (len(indexes) + 1), 6.4),
        constrained_layout=True,
    )
    for row, row_name in enumerate(("Original SAM", "M3")):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Input")
        axes[row, 0].set_ylabel(row_name, fontsize=13, fontweight="bold")
        axes[row, 0].axis("off")

    for column, block_index in enumerate(indexes, start=1):
        sam_overlay = overlay_heatmap(image, sam_maps[block_index], cmap_name)
        m3_overlay = overlay_heatmap(image, m3_maps[block_index], cmap_name)
        axes[0, column].imshow(sam_overlay)
        axes[1, column].imshow(m3_overlay)
        axes[0, column].set_title(f"Block {block_index}")
        axes[1, column].set_title(f"Block {block_index}")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
        plt.imsave(parts_dir / f"sam_block_{block_index:02d}.png", sam_overlay)
        plt.imsave(parts_dir / f"m3_block_{block_index:02d}.png", m3_overlay)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(parts_dir / "input.png")
    print(f"Saved comparison: {output.resolve()}")
    print(f"Saved individual panels: {parts_dir.resolve()}")


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def autocast_context(use_amp, device):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def main():
    args = parse_args()
    device = torch.device(args.device)
    image_path = Path(args.image)
    image_array, image_cpu = prepare_image(image_path)
    # Generate or load the DINO prior before loading ViT-H to limit peak VRAM.
    cache_dir = resolve_prior_cache_dir(args.prior_cache_dir, str(image_path.parent))
    prior_generator = LazyGroundingDinoRuntimePriorGenerator(
        blip_model_name=args.blip_model_name,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    prior_cache = PriorCache(cache_dir) if cache_dir else None
    prior = load_or_generate_prior(
        prior_generator,
        image_to_pil(image_array),
        str(image_path),
        target_text=args.manual_description or None,
        prior_cache=prior_cache,
    )
    prior_mask_cpu = prior.prior_mask.detach().cpu()
    del prior
    del prior_generator
    clear_cuda_cache()

    with torch.inference_mode():
        baseline_sam = sam_model_registry[args.model_type](
            checkpoint=args.checkpoint
        ).to(device)
        baseline_sam.eval()
        indexes = injection_indexes(baseline_sam.image_encoder)
        with autocast_context(args.use_amp, device):
            sam_features = extract_encoder_stages(
                baseline_sam.image_encoder,
                image_cpu.to(device),
                indexes,
                enhanced=False,
            )
        baseline_sam = baseline_sam.cpu()
        del baseline_sam
        clear_cuda_cache()

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
        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict, strict=True)
        del checkpoint
        del state_dict
        model = model.to(device).eval()

        with autocast_context(args.use_amp, device):
            prior_mask = prior_mask_cpu.to(device=device, dtype=image_cpu.dtype)
            if prior_mask.dim() == 3:
                prior_mask = prior_mask.unsqueeze(0)
            soft_prior = model._soften_prior_mask(prior_mask)
            prior_context = model.prior_context_encoder(soft_prior)
            m3_features = extract_encoder_stages(
                model.image_encoder,
                image_cpu.to(device),
                indexes,
                prior_context=prior_context,
                enhanced=True,
            )

    sam_maps = {}
    m3_maps = {}
    for block_index in indexes:
        sam_energy = feature_energy(sam_features[block_index])
        m3_energy = feature_energy(m3_features[block_index])
        sam_maps[block_index], m3_maps[block_index] = jointly_normalize(
            sam_energy, m3_energy
        )

    save_figure(
        args.output,
        image_array,
        indexes,
        sam_maps,
        m3_maps,
        args.colormap,
    )


if __name__ == "__main__":
    main()
