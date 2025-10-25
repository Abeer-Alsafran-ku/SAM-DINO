"""Utility helpers for fine-tuning Grounding DINO + SAM."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
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
    ) -> None:
        with open(coco_json_path, "r", encoding="utf-8") as f:
            self.coco: Dict = json.load(f)

        self.images_dir = images_dir
        self.id2img = {img["id"]: img for img in self.coco.get("images", [])}
        self.imgid_to_anns: Dict[int, List[Dict]] = {}
        for ann in self.coco.get("annotations", []):
            self.imgid_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = list(self.id2img.keys())
        self.transforms = transforms
        self.return_masks = return_masks

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_meta = self.id2img[img_id]
        img_path = self.images_dir / img_meta["file_name"]
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        anns = self.imgid_to_anns.get(img_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            if self.return_masks and "mask_path" in ann:
                mask = np.array(Image.open(ann["mask_path"]).convert("L")) > 128
                masks.append(mask.astype(np.uint8))

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

        target = {"boxes": box_tensor, "labels": label_tensor, "image_id": torch.tensor([img_id])}
        if self.return_masks:
            mask_tensor = (
                torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8)
                if masks
                else torch.zeros((0, height, width), dtype=torch.uint8)
            )
            target["masks"] = mask_tensor

        img = self.transforms(img) if self.transforms is not None else T.ToTensor()(img)
        return img, target


def collate_fn(batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Collate function compatible with the GroundingDINO training loop."""

    images, targets = zip(*batch)
    return list(images), list(targets)


def build_sam_model(checkpoint_path: Path, device: torch.device, finetune: bool) -> torch.nn.Module:
    sam_model_registry = _lazy_import_sam()
    sam_model = sam_model_registry["vit_h"](checkpoint=str(checkpoint_path)).to(device)
    if not finetune:
        for param in sam_model.parameters():
            param.requires_grad = False
    return sam_model


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

    transforms = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    train_dataset = CocoLikeDataset(
        images_dir=paths.images_dir,
        coco_json_path=paths.annotations,
        transforms=transforms,
        return_masks=return_masks,
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

    sam_model: Optional[torch.nn.Module] = None
    if paths.sam_checkpoint is not None:
        sam_model = build_sam_model(paths.sam_checkpoint, device_obj, finetune_sam)

    params = [p for p in model.parameters() if p.requires_grad]
    if sam_model is not None:
        params.extend(p for p in sam_model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device_obj.type == "cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = torch.stack(imgs).to(device_obj)
            processed_targets: List[Dict[str, torch.Tensor]] = []
            for target in targets:
                processed_target = {k: v.to(device_obj) for k, v in target.items() if k in {"boxes", "labels"}}
                if return_masks and "masks" in target:
                    processed_target["masks"] = target["masks"].to(device_obj)
                processed_targets.append(processed_target)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(imgs)
                loss_dict = criterion(outputs, processed_targets)
                loss = sum(loss_dict.values())

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
    parser.add_argument("--sam-checkpoint", type=Path, help="SAM checkpoint for joint fine-tuning")
    parser.add_argument("--output", type=Path, default=Path("./checkpoints"), help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--finetune-sam", action="store_true")
    parser.add_argument("--return-masks", action="store_true")
    parser.add_argument("--image-size", type=int, default=800, help="Resize images to this square resolution")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = TrainingPaths(
        annotations=args.annotations,
        images_dir=args.images,
        groundingdino_config=args.config,
        groundingdino_checkpoint=args.checkpoint,
        sam_checkpoint=args.sam_checkpoint,
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
