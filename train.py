# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
from datetime import datetime
import shutil
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple, Type
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial
from eval_metrics import BinarySegmentationMeter
from training_metrics import compute_best_score
from utils_downstream.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
from vlm_prior import (
    GroundingDinoRuntimePriorGenerator,
    LazyGroundingDinoRuntimePriorGenerator,
    PriorCache,
    image_to_pil,
    load_or_generate_prior,
    resolve_prior_cache_dir,
)
from vlsam_dino_model import VLSAMDino, structure_loss

torch.manual_seed(2024)
torch.cuda.empty_cache()

def _single_path(path_value):
    if isinstance(path_value, (list, tuple)):
        return path_value[0]
    return path_value


def eval_psnr(loader, model, prior_generator, prior_cache=None, eval_type=None, device=None):
    model.eval()
    
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    

    mae, sm, em, wfm = cal_mae(), cal_sm(), cal_em(device=device, compute_max=False), cal_wfm()
    fm = cal_fm(num=len(loader))
    binary_meter = BinarySegmentationMeter(threshold=0.5)


    #for batch in loader:
    for step, (image, gt2D, img_1024_ori, img_path, gt_ori) in enumerate(loader):
       
        image, gt2D = image.to(device), gt2D.to(device)
        with torch.no_grad():
            prior = load_or_generate_prior(
                prior_generator,
                image_to_pil(img_1024_ori),
                _single_path(img_path),
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
            res_t = F.interpolate(
                pred, size=gt_t.shape, mode="bilinear", align_corners=False
            ).squeeze()
            gt = gt_t.detach().cpu().numpy()
            res = res_t.detach().cpu().numpy()
            
            mae.update(res, gt)
            sm.update(res, gt)
            em.update(res_t, gt_t)
            wfm.update(res,gt)
            fm.update(res, gt)
            binary_meter.update(res, gt)
        
        if pbar is not None:
            pbar.update(1)

    MAE = mae.show()
    sm = sm.show()
    adaptive_e = em.show()
    wfm = wfm.show()
    max_f, _mean_f, _precision_curve, _recall_curve = fm.show()
    binary_metrics = binary_meter.show()
            

    if pbar is not None:
        pbar.close()

    return {
        "Sm": sm,
        "S-measure": sm,
        "alphaE": adaptive_e,
        "adaptive-E-measure": adaptive_e,
        "MAE": MAE,
        "Fomega": wfm,
        "Fx": max_f,
        "weighted-F-measure": wfm,
        "max-F-measure": max_f,
        "precision": binary_metrics["precision"],
        "recall": binary_metrics["recall"],
        "F1": binary_metrics["F1"],
    }
    
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class NpyDataset(Dataset):
    def __init__(
        self,
        data_root,
        bbox_shift=20,
        img_dir="Imgs/",
        gt_dir="GT/",
        return_original_gt=False,
    ):
        self.data_root = data_root
        self.gt_path = join(data_root, gt_dir)
        self.img_path = join(data_root, img_dir)
        self.bbox_shift = bbox_shift
        self.return_original_gt = return_original_gt

        img_files = [
            f
            for f in os.listdir(self.img_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        img_stem_map = {os.path.splitext(f)[0]: join(self.img_path, f) for f in img_files}
        gt_files = sorted(f for f in os.listdir(self.gt_path) if f.endswith('.png'))
        self.samples = []
        for gt_f in gt_files:
            stem = os.path.splitext(gt_f)[0]
            if stem in img_stem_map:
                self.samples.append((join(self.gt_path, gt_f), img_stem_map[stem]))
        print(f"number of images: {len(self.samples)}")
        
        self.img_transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.mask_transform = transforms.Compose([
                transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        gt_path, img_path = self.samples[index]
        img_1024_ori = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        gt_ori = np.asarray(gt, dtype=np.float32) / 255.0

        img_1024 = self.img_transform(img_1024_ori)
        gt = self.mask_transform(gt)

        sample = (
            img_1024,
            gt,
            np.array(img_1024_ori),
            img_path,
        )
        if self.return_original_gt:
            return sample + (gt_ori,)
        return sample

# %% set up parser
parser = argparse.ArgumentParser()
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
parser.add_argument("-task_name", type=str, default="camo")
parser.add_argument("-model_type", type=str, default="vit_h")
parser.add_argument(
    "-checkpoint", type=str, default="work_dir/SAM/sam_vit_h_4b8939.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0002, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--blip_model_name", type=str, default=GroundingDinoRuntimePriorGenerator.DEFAULT_BLIP_MODEL)
parser.add_argument("--dino_model_name", type=str, default=GroundingDinoRuntimePriorGenerator.DEFAULT_DINO_MODEL)
parser.add_argument("--dino_text_dim", type=int, default=256)
parser.add_argument("--prior_cache_dir", type=str, default=None)
parser.add_argument("--prior_dropout_prob", type=float, default=0.0)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()
args.prior_cache_dir = resolve_prior_cache_dir(args.prior_cache_dir, args.tr_npy_path)

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name)
device = torch.device(args.device)
def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    vlsam_model = VLSAMDino(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        text_feature_dim=args.dino_text_dim,
        prior_dropout_prob=args.prior_dropout_prob,
    ).to(device)

    for param in vlsam_model.image_encoder.parameters():
        param.requires_grad = False
    if hasattr(vlsam_model.image_encoder, "adapters_enabled"):
        vlsam_model.image_encoder.adapters_enabled = True
    for name, param in vlsam_model.image_encoder.named_parameters():
        if (
            "adapters" in name
            or "adapter_gamma" in name
            or "prior_injector" in name
            or "prior_gamma" in name
        ):
            param.requires_grad = True

    vlsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in vlsam_model.parameters()),
    )  # 93735472
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in vlsam_model.parameters() if p.requires_grad),
    )  # 93729252

    img_mask_encdec_params = list(vlsam_model.image_encoder.parameters()) + list(
        vlsam_model.mask_decoder.parameters()
    )

    encoder_params = list(vlsam_model.image_encoder.parameters())
    decoder_params = list(vlsam_model.mask_decoder.parameters())
    prior_params = list(vlsam_model.text_proj.parameters()) + list(
        vlsam_model.prior_context_encoder.parameters()
    ) + list(
        vlsam_model.prior_mask_encoder.parameters()
    )

    # Create parameter groups
    
    param_groups = [
    {'params': encoder_params, 'lr': args.lr * 0.1},
    {'params': decoder_params, 'lr': args.lr},
    {'params': prior_params, 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    
    lr_scheduler = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1.0e-6)
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  # 93729252
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    best_loss = 1e10
    best_accuracy =0
    train_dataset = NpyDataset(
        args.tr_npy_path,
        img_dir=args.train_img_dir,
        gt_dir=args.train_gt_dir,
    )

    test_dataset = NpyDataset(
        args.tr_npy_path,
        img_dir=args.test_img_dir,
        gt_dir=args.test_gt_dir,
        return_original_gt=True,
    )
    

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            vlsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    prior_generator = LazyGroundingDinoRuntimePriorGenerator(
        blip_model_name=args.blip_model_name,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    prior_cache = PriorCache(args.prior_cache_dir) if args.prior_cache_dir else None
    print("Prior cache directory:", args.prior_cache_dir)
    print("BLIP/DINO will load only after a cache miss.")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_structure_loss = 0
        epoch_accuracy = 0
        
        for step, (image, gt2D,img_1024_ori,img_path) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
                    
            image, gt2D = image.to(device), gt2D.to(device)
            with torch.no_grad():
               prior = load_or_generate_prior(
                   prior_generator,
                   image_to_pil(img_1024_ori),
                   _single_path(img_path),
                   prior_cache=prior_cache,
               )
               prior_text_features = prior.text_features.to(device)
               prior_mask = prior.prior_mask.to(device)
               prior_confidence = prior.confidence_tensor.to(device)

            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = vlsam_model(
                        image,
                        prior_text_features,
                        prior_mask,
                        prior_confidence,
                    )
                    structure_loss_value = structure_loss(medsam_pred, gt2D.float())
                    loss = structure_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = vlsam_model(
                    image,
                    prior_text_features,
                    prior_mask,
                    prior_confidence,
                )
                structure_loss_value = structure_loss(medsam_pred, gt2D.float())
                loss = structure_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
       
            epoch_loss += loss.item()
            epoch_structure_loss += structure_loss_value.detach().item()
            iter_num += 1
        
        val_metrics = eval_psnr(
            test_dataloader,
            vlsam_model,
            prior_generator,
            prior_cache=prior_cache,
            eval_type='cod',
            device=device,
        )
        print({'Sm': val_metrics["Sm"]})
        print({'alphaE': val_metrics["alphaE"]})
        print({'MAE': val_metrics["MAE"]})
        print({'Fomega': val_metrics["Fomega"]})
        print({'Fx': val_metrics["Fx"]})
        print({'precision': val_metrics["precision"]})
        print({'recall': val_metrics["recall"]})
        print({'F1': val_metrics["F1"]})
        print('Prior cache:', prior_cache.summary(), 'runtime_loaded=', prior_generator.loaded)
        vlsam_model.train()

        epoch_loss /= (step + 1)
        epoch_structure_loss /= (step + 1)
        lr_scheduler.step()
        epoch_accuracy = compute_best_score(val_metrics)
        losses.append(epoch_loss)
        if args.use_wandb:
            wandb.log(
                {
                    "epoch_loss": epoch_loss,
                    "structure_loss": epoch_structure_loss,
                }
            )
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        print(
            {
                "structure_loss": epoch_structure_loss,
            }
        )
        ## save the latest model
        checkpoint = {
            "model": vlsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "vlsam_model_latest.pth"))
        ## save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            checkpoint = {
                "model": vlsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "vlsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Structure Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()

