# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
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
from PIL import Image
from torchvision import transforms
from typing import Any, Optional, Tuple
import torch
from utils_downstream.saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm, cal_dice, cal_iou,cal_ber,cal_acc
from vlm_prior import (
    GroundingDinoRuntimePriorGenerator,
    LazyGroundingDinoRuntimePriorGenerator,
    PriorCache,
    image_to_pil,
    load_or_generate_prior,
    resolve_prior_cache_dir,
)
from vlsam_dino_model import VLSAMDino
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

DEFAULT_IMAGE_PATH = "E:\閻柧绔碶閻柧绨‐閺佺増宓乗VOC-USC12K\426.jpg"
DEFAULT_GT_PATH = ""
MANUAL_DESCRIPTION = ""


def save_prediction_mask(prediction, img_path, save_dir):
    if isinstance(img_path, (list, tuple)):
        img_path = img_path[0]

    mask = np.clip(prediction * 255, 0, 255).astype(np.uint8)
    mask_img = Image.fromarray(mask)

    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    with Image.open(img_path) as ori_img:
        mask_img = mask_img.resize(ori_img.size, resample)

    save_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
    mask_img.save(join(save_dir, save_name))


def _single_path(path_value):
    if isinstance(path_value, (list, tuple)):
        return path_value[0]
    return path_value


def eval_psnr(loader, model,prior_generator,manual_description,eval_type=None,device=None,save_dir=None,prior_cache=None):
    model.eval()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
      
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    
    mae,sm,em,wfm, m_dice, m_iou,ber,acc= cal_mae(),cal_sm(),cal_em(),cal_wfm(), cal_dice(), cal_iou(),cal_ber(),cal_acc()

    for step, (image, gt2D,img_1024_ori,img_path,has_gt) in enumerate(loader):

        image, gt2D = image.to(device), gt2D.to(device)
        with torch.no_grad():
            prior = load_or_generate_prior(
                prior_generator,
                image_to_pil(img_1024_ori),
                _single_path(img_path),
                target_text=manual_description or None,
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
            
            res = pred.squeeze().squeeze().cpu().numpy()
            gt = gt2D.squeeze().squeeze().cpu().numpy()

            has_gt_value = bool(has_gt[0].item()) if torch.is_tensor(has_gt) else bool(has_gt[0])
            if has_gt_value:
                mae.update(res, gt)
                sm.update(res,gt)
                em.update(res,gt)
                wfm.update(res,gt)
                m_dice.update(res,gt)
                m_iou.update(res,gt)
                ber.update(res,gt)

            if save_dir is not None:
                save_prediction_mask(res, img_path, save_dir)
        
        if pbar is not None:
            pbar.update(1)

    if len(mae.prediction) == 0:
        MAE, sm, em, wfm = None, None, None, None
    else:
        MAE = mae.show()
        sm = sm.show()
        em = em.show()
        wfm = wfm.show()
        m_dice = m_dice.show()
        m_iou = m_iou.show()
        ber = ber.show()

    if pbar is not None:
        pbar.close()

    return sm, em, wfm, MAE
    
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
    def __init__(self, data_root, bbox_shift=20, image_path="", gt_path=""):
        self.data_root = data_root
        self.gt_path = join(data_root, "GT/")
        self.img_path = join(data_root, "Imgs/")
        if image_path:
            self.img_path_files = [image_path]
            self.gt_path_files = [gt_path or self._find_gt_path(image_path)]
        else:
            self.gt_path_files = sorted([self.gt_path + f for f in os.listdir(self.gt_path) if f.endswith('.png')])
            self.img_path_files = sorted([self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg')])
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.img_path_files)}")
        
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

    def _find_gt_path(self, image_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        for ext in (".png", ".jpg", ".bmp"):
            gt_path = join(self.gt_path, image_name + ext)
            if os.path.exists(gt_path):
                return gt_path
        return ""

    def __len__(self):
        return len(self.img_path_files)

    def __getitem__(self, index):

        img_1024_ori = Image.open(self.img_path_files[index]).convert('RGB')
        has_gt = bool(self.gt_path_files[index]) and os.path.exists(self.gt_path_files[index])
        if has_gt:
            gt = Image.open(self.gt_path_files[index]).convert('L')  # multiple labels [0, 1,4,5...], (256,256)
        else:
            gt = Image.new('L', img_1024_ori.size, 0)
        
        img_1024 = self.img_transform(img_1024_ori)
        gt = self.mask_transform(gt)
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt).long(),
            np.array(img_1024_ori),
            self.img_path_files[index],
            has_gt,
        )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/TrainDataset",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="./work_dir/SOTA_real/vlsam_model_best.pth"
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="use wandb to monitor training"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
parser.add_argument("--result_dir", type=str, default="./work_dir/results/CAMO")
parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH)
parser.add_argument("--gt_path", type=str, default=DEFAULT_GT_PATH)
parser.add_argument("--manual_description", type=str, default=MANUAL_DESCRIPTION)
parser.add_argument("--blip_model_name", type=str, default=GroundingDinoRuntimePriorGenerator.DEFAULT_BLIP_MODEL)
parser.add_argument("--dino_model_name", type=str, default=GroundingDinoRuntimePriorGenerator.DEFAULT_DINO_MODEL)
parser.add_argument("--dino_text_dim", type=int, default=256)
parser.add_argument("--prior_cache_dir", type=str, default=None)

parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()
args.prior_cache_dir = resolve_prior_cache_dir(
    args.prior_cache_dir,
    args.image_path or args.tr_npy_path,
)


# %% set up model for training
# device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
def main():
    if not args.image_path:
        raise ValueError("Please provide --image_path for single-image inference.")

    sam_model = sam_model_registry[args.model_type](checkpoint='work_dir/SAM/sam_vit_b_01ec64.pth')
    vlsam_model = VLSAMDino(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        text_feature_dim=args.dino_text_dim,
        
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    vlsam_model.load_state_dict(state_dict)
    
    test_dataset = NpyDataset('data/TestDataset/CAMO', image_path=args.image_path, gt_path=args.gt_path)
    print("Number of training samples: ", len(test_dataset))

    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        #num_workers=args.num_workers,
        pin_memory=True,
    )

    prior_generator = LazyGroundingDinoRuntimePriorGenerator(
        blip_model_name=args.blip_model_name,
        dino_model_name=args.dino_model_name,
        device=device,
    )
    prior_cache = PriorCache(args.prior_cache_dir) if args.prior_cache_dir else None
    
            
    result1, result2, result3, result4 = eval_psnr(test_dataloader, vlsam_model,prior_generator,args.manual_description,
                eval_type='cod',device=device,save_dir=args.result_dir,prior_cache=prior_cache)
    if result1 is not None:
        print({'Sm': result1})
        print({'Em': result2})
        print({'wFm': result3})
        print({'Mae': result4})
    else:
        print("No GT mask was found; skipped metric calculation.")
    print("Manual description:", args.manual_description)
    print("Saved prediction masks to:", args.result_dir)
    print("Prior cache:", prior_cache.summary(), "runtime_loaded=", prior_generator.loaded)




if __name__ == "__main__":
    main()
