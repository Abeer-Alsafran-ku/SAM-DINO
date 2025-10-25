"""Utility helpers for fine-tuning Grounding DINO + SAM."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


@dataclass
class TrainingPaths:
    """Container describing the resources required for training."""

    annotations: Path
    images_dir: Path
    groundingdino_config: Path
    groundingdino_checkpoint: Optional[Path] = None
    sam_checkpoint: Optional[Path] = None
    sam_model_type: str = "vit_h"
    output_dir: Path = Path("./checkpoints")


def _lazy_import_groundingdino():
    """Import GroundingDINO modules with a friendlier error message."""

    try:
        from groundingdino.models import build_model  # type: ignore[import]
        from groundingdino.models.matcher import build_matcher  # type: ignore[import]
        from groundingdino.models.criterion import SetCriterion  # type: ignore[import]
        from groundingdino.util.misc import clean_state_dict  # type: ignore[import]
        from groundingdino.util.slconfig import SLConfig  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - provides friendlier message in production
        raise ImportError(
            "GroundingDINO is not available. Make sure the package is installed and the "
            "repository root is added to PYTHONPATH. On Colab you can run:"
            "\n  !pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
            "\nand then restart the kernel before re-running this script."
        ) from exc

    return build_model, build_matcher, SetCriterion, clean_state_dict, SLConfig


def _lazy_import_sam():
    try:
        from segment_anything import sam_model_registry  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "segment-anything is not available. Install it with"
            "\n  !pip install git+https://github.com/facebookresearch/segment-anything.git"
            "\nand restart the runtime before continuing."
        ) from exc

    return sam_model_registry


def _lazy_import_coco_mask_utils():
    try:
        from pycocotools import mask as mask_utils  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pycocotools is required for mask loading. Install it with"
            "\n  pip install pycocotools"
            "\nand rerun the script."
        ) from exc

    return mask_utils


def load_groundingdino_config(config_path: Path):
    """Load a GroundingDINO configuration file."""

    _, _, _, _, SLConfig = _lazy_import_groundingdino()
    config = SLConfig.fromfile(str(config_path))
    return config


def build_groundingdino_model_and_criterion(
    config_path: Path,
    checkpoint_path: Optional[Path],
    device: torch.device,
):
    """Instantiate Grounding DINO and its SetCriterion matcher/loss."""

    (
        build_model,
        build_matcher,
        SetCriterion,
        clean_state_dict,
        SLConfig,
    ) = _lazy_import_groundingdino()

    config = SLConfig.fromfile(str(config_path))
    model = build_model(config)

    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():  # pragma: no cover - defensive runtime check
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")
        state_dict = clean_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    matcher = build_matcher(config)
    weight_dict = getattr(config.MODEL, "LOSS", None)
    if weight_dict is None:
        # Default weights from the official config
        weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}

    num_classes = getattr(config.MODEL, "NUM_CLASSES", None)
    if num_classes is None:
        num_classes = getattr(getattr(config.MODEL, "DINO", object()), "NUM_CLASSES", None)
    if num_classes is None:
        raise AttributeError(
            "Unable to infer the number of classes from the config. "
            "Please set MODEL.NUM_CLASSES or MODEL.DINO.NUM_CLASSES in the config."
        )

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=getattr(config.MODEL, "EOS_COEF", 0.1),
    )
    criterion.to(device)

    return model, criterion


class CocoLikeDataset(Dataset):
    """Minimal COCO-style dataset wrapper used during fine-tuning."""

    def __init__(
        self,
        images_dir: Path,
        coco_json_path: Path,
        transforms: Optional[T.Compose] = None,
        return_masks: bool = False,
        image_size: Optional[int] = None,
    ) -> None:
        with open(coco_json_path, "r", encoding="utf-8") as f:
            self.coco: Dict = json.load(f)

        self.images_dir = images_dir
        self.id2img = {img["id"]: img for img in self.coco.get("images", [])}
        self.imgid_to_anns: Dict[int, List[Dict]] = {}
        for ann in self.coco.get("annotations", []):
            self.imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

        if not self.id2img:
            raise ValueError(
                "No images found in the COCO json. Ensure the file follows the COCO schema."
            )

        self.image_ids = list(self.id2img.keys())
        self.transforms = transforms
        self.return_masks = return_masks
        self.image_size = image_size

        categories = self.coco.get("categories", [])
        if categories:
            sorted_categories = sorted(categories, key=lambda cat: cat["id"])
            self.cat_id_to_contig = {cat["id"]: idx for idx, cat in enumerate(sorted_categories)}
        else:
            ann_category_ids = sorted(
                {ann["category_id"] for ann in self.coco.get("annotations", [])}
            )
            if not ann_category_ids:
                raise ValueError(
                    "Unable to infer categories from the dataset. Provide a COCO 'categories' section."
                )
            self.cat_id_to_contig = {cat_id: idx for idx, cat_id in enumerate(ann_category_ids)}
        self.contig_to_cat_id = {v: k for k, v in self.cat_id_to_contig.items()}
        self.num_categories: int = len(self.cat_id_to_contig)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_meta = self.id2img[img_id]
        img_path = self.images_dir / img_meta["file_name"]
        img = Image.open(img_path).convert("RGB")
        orig_width, orig_height = img.size

        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            resize_width, resize_height = self.image_size, self.image_size
            scale_x = resize_width / orig_width
            scale_y = resize_height / orig_height
        else:
            resize_width, resize_height = orig_width, orig_height
            scale_x = scale_y = 1.0

        anns = self.imgid_to_anns.get(img_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        mask_boxes: List[List[float]] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1 = np.clip(x * scale_x, 0, resize_width)
            y1 = np.clip(y * scale_y, 0, resize_height)
            x2 = np.clip((x + bw) * scale_x, 0, resize_width)
            y2 = np.clip((y + bh) * scale_y, 0, resize_height)
            if x2 <= x1 or y2 <= y1:
                warnings.warn(
                    f"Skipping invalid bbox {ann['bbox']} for image_id={img_id}",
                    RuntimeWarning,
                )
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_contig.get(ann["category_id"], 0))
            if self.return_masks:
                mask = self._load_mask(ann, orig_height, orig_width)
                if mask is not None:
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    if self.image_size is not None:
                        mask_img = mask_img.resize((resize_width, resize_height), Image.NEAREST)
                    masks.append((np.array(mask_img) > 128).astype(np.uint8))
                    mask_boxes.append([x1, y1, x2, y2])

        box_tensor = (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        label_tensor = (
            torch.tensor(labels, dtype=torch.int64)
            if labels
            else torch.zeros((0,), dtype=torch.int64)
        )

        target: Dict[str, torch.Tensor] = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "orig_size": torch.tensor([orig_height, orig_width], dtype=torch.int64),
            "size": torch.tensor([resize_height, resize_width], dtype=torch.int64),
        }
        if self.return_masks:
            mask_tensor = (
                torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8)
                if masks
                else torch.zeros((0, resize_height, resize_width), dtype=torch.uint8)
            )
            target["masks"] = mask_tensor
            mask_box_tensor = (
                torch.tensor(mask_boxes, dtype=torch.float32)
                if mask_boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            target["mask_boxes"] = mask_box_tensor

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)
        else:
            img_tensor = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(img_tensor)

        return img_tensor, target

    def _load_mask(self, ann: Dict, height: int, width: int) -> Optional[np.ndarray]:
        if "segmentation" in ann and ann["segmentation"]:
            mask_utils = _lazy_import_coco_mask_utils()
            segmentation = ann["segmentation"]
            if isinstance(segmentation, list):
                rles = mask_utils.frPyObjects(segmentation, height, width)
                rle = mask_utils.merge(rles)
            elif isinstance(segmentation, dict):
                rle = segmentation
            else:
                raise TypeError(f"Unsupported segmentation format: {type(segmentation)}")
            mask = mask_utils.decode(rle)
            return mask.astype(np.uint8)

        if "mask_path" in ann:
            mask_path = self.images_dir / ann["mask_path"]
            if mask_path.exists():
                return (np.array(Image.open(mask_path).convert("L")) > 128).astype(np.uint8)
            warnings.warn(f"Mask path {mask_path} not found; ignoring mask.", RuntimeWarning)

        return None


def collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Collate function compatible with the GroundingDINO training loop."""

    images, targets = zip(*batch)
    return list(images), list(targets)


def build_sam_model(
    checkpoint_path: Path,
    model_type: str,
    device: torch.device,
    finetune: bool,
) -> torch.nn.Module:
    sam_model_registry = _lazy_import_sam()
    if model_type not in sam_model_registry:
        available = ", ".join(sorted(sam_model_registry.keys()))
        raise ValueError(
            f"Unknown SAM model type '{model_type}'. Available options are: {available}"
        )

    sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(device)
    sam_model.train(mode=finetune)
    if not finetune:
        for param in sam_model.parameters():
            param.requires_grad = False
    return sam_model


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inputs = torch.sigmoid(inputs)
    inputs = inputs.view(inputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (inputs * targets).sum(dim=1)
    union = inputs.sum(dim=1) + targets.sum(dim=1)
    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss.mean()


def compute_sam_mask_loss(
    sam_model: torch.nn.Module,
    images: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
) -> Optional[torch.Tensor]:
    total_losses: List[torch.Tensor] = []
    for img_tensor, target in zip(images, targets):
        if "masks" not in target or target["masks"].numel() == 0:
            continue
        if target["boxes"].numel() == 0:
            continue

        image_batch = img_tensor.unsqueeze(0)
        preprocessed = sam_model.preprocess(image_batch)
        image_embeddings = sam_model.image_encoder(preprocessed)
        dense_pe = sam_model.prompt_encoder.get_dense_pe().to(image_embeddings.device)

        boxes = target.get("mask_boxes", target["boxes"])
        if boxes.numel() == 0:
            continue
        boxes = boxes.unsqueeze(0)
        transformed_boxes = sam_model.prompt_encoder.box_transform.apply_boxes_torch(
            boxes, preprocessed.shape[-2:]
        )
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=transformed_boxes,
            masks=None,
        )
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        upscaled_masks = sam_model.postprocess_masks(
            low_res_masks,
            input_size=preprocessed.shape[-2:],
            original_size=image_batch.shape[-2:],
        )
        pred_masks = upscaled_masks.squeeze(1)
        gt_masks = target["masks"].float()
        if gt_masks.shape[-2:] != pred_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.unsqueeze(0),
                size=pred_masks.shape[-2:],
                mode="nearest",
            ).squeeze(0)

        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        dice = dice_loss(pred_masks, gt_masks)
        total_losses.append(bce + dice)

    if not total_losses:
        return None

    return torch.stack(total_losses).mean()


def train(
    paths: TrainingPaths,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    finetune_sam: bool = False,
    return_masks: bool = False,
    image_size: int = 800,
):
    """Fine-tune Grounding DINO (optionally together with SAM) on a COCO-style dataset."""

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    if device_obj.type != device:
        print(f"Requested device '{device}' is not available. Falling back to {device_obj}.")

    os.makedirs(paths.output_dir, exist_ok=True)
    print("~~~~~~~~~ Started Training ~~~~~~~~~")

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = CocoLikeDataset(
        images_dir=paths.images_dir,
        coco_json_path=paths.annotations,
        transforms=normalize,
        return_masks=return_masks,
        image_size=image_size,
    )
    print(
        f"Loaded {len(train_dataset)} training samples spanning {train_dataset.num_categories} categories."
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model, criterion = build_groundingdino_model_and_criterion(
        config_path=paths.groundingdino_config,
        checkpoint_path=paths.groundingdino_checkpoint,
        device=device_obj,
    )

    if finetune_sam and paths.sam_checkpoint is None:
        raise ValueError("--finetune-sam requires --sam-checkpoint to be provided.")

    sam_model: Optional[torch.nn.Module] = None
    if paths.sam_checkpoint is not None:
        if finetune_sam and not return_masks:
            raise ValueError("--finetune-sam requires --return-masks to provide supervision.")
        sam_model = build_sam_model(
            checkpoint_path=paths.sam_checkpoint,
            model_type=paths.sam_model_type,
            device=device_obj,
            finetune=finetune_sam,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    if sam_model is not None and any(p.requires_grad for p in sam_model.parameters()):
        params.extend(p for p in sam_model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device_obj.type == "cuda")

    for epoch in range(epochs):
        model.train()
        if sam_model is not None:
            sam_model.train(mode=finetune_sam)
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = torch.stack(imgs).to(device_obj)
            processed_targets: List[Dict[str, torch.Tensor]] = []
            for target in targets:
                processed_target = {
                    key: value.to(device_obj)
                    for key, value in target.items()
                    if key in {"boxes", "labels", "masks", "orig_size", "size", "mask_boxes"}
                }
                processed_targets.append(processed_target)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(imgs)
                detection_targets = [
                    {k: v for k, v in tgt.items() if k != "mask_boxes"}
                    for tgt in processed_targets
                ]
                loss_dict = criterion(outputs, detection_targets)
                loss = sum(loss_dict.values())
                if sam_model is not None and return_masks:
                    mask_loss = compute_sam_mask_loss(sam_model, imgs, processed_targets)
                    if mask_loss is not None:
                        loss = loss + mask_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1}/{epochs}, avg_loss={avg_loss:.4f}")

        ckpt_path = paths.output_dir / f"finetuned_epoch{epoch + 1}.pth"
        torch.save(
            {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    print(f"Training complete. Checkpoints saved in {paths.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GroundingDINO (optionally with SAM)")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO-style annotation json")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--config", type=Path, required=True, help="GroundingDINO config file")
    parser.add_argument("--checkpoint", type=Path, help="GroundingDINO checkpoint for initialization")
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="SAM checkpoint for joint fine-tuning",
    )
    parser.add_argument(
        "--sam-backbone",
        "--sam-model-type",
        dest="sam_backbone",
        type=str,
        default="vit_h",
        help="SAM backbone to load (e.g. vit_h, vit_l, vit_b)",
    )
    parser.add_argument(
        "--sam-model",
        dest="sam_model_legacy",
        type=str,
        help="Deprecated flag kept for backwards compatibility. Use `--sam-backbone` or `--sam-checkpoint` instead.",
    )
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"), help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--finetune-sam", action="store_true")
    parser.add_argument("--return-masks", action="store_true")
    parser.add_argument("--image-size", type=int, default=800, help="Resize images to this square resolution")
    args = parser.parse_args()

    if getattr(args, "sam_model_legacy", None) is not None:
        legacy_value = args.sam_model_legacy
        legacy_path = Path(legacy_value).expanduser()
        if args.sam_checkpoint is None and (legacy_path.suffix in {".pth", ".pt"} or legacy_path.exists()):
            warnings.warn(
                "`--sam-model` is deprecated for providing checkpoint paths. "
                "Use `--sam-checkpoint` instead.",
                stacklevel=2,
            )
            args.sam_checkpoint = legacy_path
        else:
            warnings.warn(
                "`--sam-model` is deprecated for selecting the SAM backbone. "
                "Use `--sam-backbone` (or `--sam-model-type`) instead.",
                stacklevel=2,
            )
            args.sam_backbone = legacy_value

    args.sam_model_type = args.sam_backbone

    return args


def main():
    args = parse_args()
    paths = TrainingPaths(
        annotations=args.annotations,
        images_dir=args.images,
        groundingdino_config=args.config,
        groundingdino_checkpoint=args.checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        output_dir=args.output,
    )

    train(
        paths=paths,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        finetune_sam=args.finetune_sam,
        return_masks=args.return_masks,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
