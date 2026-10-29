# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from eval_metrics import BinarySegmentationMeter
from segment_anything import sam_model_registry
from utils_downstream.saliency_metric import (
    cal_acc,
    cal_dice,
    cal_em,
    cal_fm,
    cal_iou,
    cal_mae,
    cal_sm,
    cal_wfm,
)
from vlm_prior import (
    GroundingDinoRuntimePriorGenerator,
    LazyGroundingDinoRuntimePriorGenerator,
    PriorCache,
    image_to_pil,
    load_or_generate_prior,
    resolve_prior_cache_dir,
)
from vlsam_dino_model import VLSAMDino


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_image_gt_pairs(image_dir, gt_dir):
    image_dir = Path(image_dir)
    gt_dir = Path(gt_dir)
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        gt_path = _find_gt_for_image(image_path, gt_dir)
        if gt_path is not None:
            pairs.append((str(image_path), str(gt_path)))
    return pairs


def _find_gt_for_image(image_path: Path, gt_dir: Path):
    for ext in GT_EXTENSIONS:
        gt_path = gt_dir / f"{image_path.stem}{ext}"
        if gt_path.exists():
            return gt_path
    return None


def save_prediction_mask(prediction, image_path, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    mask = np.clip(prediction * 255, 0, 255).astype(np.uint8)
    mask_img = Image.fromarray(mask)
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    with Image.open(image_path) as ori_img:
        mask_img = mask_img.resize(ori_img.size, resample)
    mask_img.save(save_dir / f"{Path(image_path).stem}.png")


def resize_prediction_to_gt(prediction, gt_shape):
    height, width = gt_shape
    prediction_img = Image.fromarray(np.asarray(prediction, dtype=np.float32), mode="F")
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    return np.asarray(prediction_img.resize((width, height), resample), dtype=np.float32)


class EvalDataset(Dataset):
    def __init__(self, image_dir, gt_dir):
        self.pairs = find_image_gt_pairs(image_dir, gt_dir)
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, gt_path = self.pairs[index]
        image_ori = Image.open(image_path).convert("RGB")
        gt = Image.open(gt_path).convert("L")
        gt_ori = np.asarray(gt, dtype=np.float32) / 255.0
        return (
            self.img_transform(image_ori).float(),
            self.mask_transform(gt).float(),
            np.array(image_ori),
            image_path,
            gt_path,
            gt_ori,
        )


def _single_path(path_value):
    if isinstance(path_value, (list, tuple)):
        return path_value[0]
    return path_value


def evaluate(args):
    device = torch.device(args.device)
    dataset = EvalDataset(args.image_dir, args.gt_dir)
    if len(dataset) == 0:
        raise ValueError("No image/GT pairs found. Check --image_dir and --gt_dir.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    model = VLSAMDino(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        text_feature_dim=args.dino_text_dim,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    prior_generator = LazyGroundingDinoRuntimePriorGenerator(
        blip_model_name=args.blip_model_name,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    prior_cache = PriorCache(args.prior_cache_dir) if args.prior_cache_dir else None

    result_dir = Path(args.result_dir)
    pred_dir = result_dir / "pred_masks"
    mae, sm, em, wfm = cal_mae(), cal_sm(), cal_em(device=device), cal_wfm()
    fm = cal_fm(num=len(dataset))
    dice, iou, acc = cal_dice(), cal_iou(), cal_acc()
    binary_meter = BinarySegmentationMeter(threshold=args.threshold)

    with torch.inference_mode():
        for image, gt, image_ori, image_path, _gt_path, gt_ori in tqdm(loader, desc="eval"):
            image = image.to(device)
            gt = gt.to(device)
            image_path_value = _single_path(image_path)
            prior = load_or_generate_prior(
                prior_generator,
                image_to_pil(image_ori),
                image_path_value,
                target_text=args.manual_description or None,
                prior_cache=prior_cache,
            )
            pred = torch.sigmoid(
                model(
                    image,
                    prior.text_features.to(device),
                    prior.prior_mask.to(device),
                    prior.confidence_tensor.to(device),
                )
            )

            gt_t = gt_ori.to(device).squeeze()
            pred_t_for_em = F.interpolate(
                pred, size=gt_t.shape, mode="bilinear", align_corners=False
            ).squeeze()
            pred_np = pred.squeeze().detach().cpu().numpy()
            gt_np = gt_t.detach().cpu().numpy()
            pred_np = resize_prediction_to_gt(pred_np, gt_np.shape)
            mae.update(pred_np, gt_np)
            sm.update(pred_np, gt_np)
            em.update(pred_t_for_em, gt_t)
            wfm.update(pred_np, gt_np)
            fm.update(pred_np, gt_np)
            dice.update(pred_np, gt_np)
            iou.update(pred_np, gt_np)
            acc.update(pred_np, gt_np)
            binary_meter.update(pred_np, gt_np)
            save_prediction_mask(pred_np, image_path_value, pred_dir)

    max_f, mean_f, _precision_curve, _recall_curve = fm.show()
    adaptive_e = em.show()
    max_e = em.show_max()
    weighted_f = wfm.show()
    s_measure = sm.show()
    metrics = {
        "Sm": float(s_measure),
        "alphaE": float(adaptive_e),
        "Ex": float(max_e),
        "Fomega": float(weighted_f),
        "Fx": float(max_f),
        "adaptive-E-measure": float(adaptive_e),
        "max-E-measure": float(max_e),
        "MAE": float(mae.show()),
        "S-measure": float(s_measure),
        "E-measure": float(adaptive_e),
        "F-measure": float(mean_f),
        "max-F-measure": float(max_f),
        "weighted-F-measure": float(weighted_f),
        "Dice": float(dice.show()),
        "IoU": float(iou.show()),
        "accuracy": float(acc.show()),
    }
    metrics.update({k: float(v) for k, v in binary_meter.show().items()})
    save_metrics(metrics, result_dir)
    print("Prior cache:", prior_cache.summary(), "runtime_loaded=", prior_generator.loaded)
    return metrics


def save_metrics(metrics, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (result_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--result_dir", type=str, default="work_dir/eval_results/CAMO")
    parser.add_argument("--prior_cache_dir", type=str, default=None)
    parser.add_argument("--manual_description", type=str, default="")
    parser.add_argument("--blip_model_name", type=str, default=GroundingDinoRuntimePriorGenerator.DEFAULT_BLIP_MODEL)
    parser.add_argument("--dino_model_name", type=str, default=GroundingDinoRuntimePriorGenerator.DEFAULT_DINO_MODEL)
    parser.add_argument("--dino_text_dim", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    args.prior_cache_dir = resolve_prior_cache_dir(args.prior_cache_dir, args.image_dir)
    return args


def main():
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
