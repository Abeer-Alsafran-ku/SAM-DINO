"""Compare ground truth and predicted annotations using detection metrics.

This script loads two COCO-style annotation files (ground truth and predictions)
and computes overall precision, recall, and mean Average Precision (mAP) at an
IoU threshold of 0.5. Matching between ground truth and predicted boxes is
performed strictly by image file name to avoid mismatches when image identifiers
differ between the two files.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert


def read_coco_annotations(coco_annotation_path: Path) -> dict:
    """Load a COCO annotation file from ``coco_annotation_path``."""
    with coco_annotation_path.open("r") as f:
        return json.load(f)


def _group_annotations_by_filename(
    coco_data: dict,
) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
    """Group annotations by image filename.

    Returns a tuple ``(annotations_by_filename, images_by_filename)`` where:

    * ``annotations_by_filename`` maps file names to lists of annotation dicts.
    * ``images_by_filename`` maps file names to the image dicts themselves.
    """

    images_by_id = {image["id"]: image for image in coco_data.get("images", [])}
    annotations_by_filename: Dict[str, List[dict]] = defaultdict(list)
    images_by_filename: Dict[str, dict] = {}

    for image_id, image in images_by_id.items():
        file_name = image.get("file_name")
        if file_name is not None:
            images_by_filename[file_name] = image

    for annotation in coco_data.get("annotations", []):
        image_id = annotation.get("image_id")
        image = images_by_id.get(image_id)
        if not image:
            # Annotation without a corresponding image entry – skip it.
            continue
        file_name = image.get("file_name")
        if file_name is None:
            continue
        annotations_by_filename[file_name].append(annotation)

    return annotations_by_filename, images_by_filename


def _prepare_ground_truth(
    annotations_by_filename: Dict[str, List[dict]]
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, torch.Tensor]]]:
    """Convert ground truth boxes to tensors grouped by file name and category."""

    gt_boxes: Dict[str, Dict[int, torch.Tensor]] = {}
    matched_flags: Dict[str, Dict[int, torch.Tensor]] = {}

    for file_name, annotations in annotations_by_filename.items():
        boxes_by_category: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for ann in annotations:
            bbox = ann.get("bbox")
            category_id = ann.get("category_id")
            if bbox is None or category_id is None:
                continue
            bbox_tensor = box_convert(
                torch.tensor(bbox, dtype=torch.float32), "xywh", "xyxy"
            )
            boxes_by_category[category_id].append(bbox_tensor)

        gt_boxes[file_name] = {}
        matched_flags[file_name] = {}
        for category_id, boxes in boxes_by_category.items():
            if boxes:
                stacked = torch.stack(boxes, dim=0)
            else:
                stacked = torch.empty((0, 4), dtype=torch.float32)
            gt_boxes[file_name][category_id] = stacked
            matched_flags[file_name][category_id] = torch.zeros(
                stacked.shape[0], dtype=torch.bool
            )

    return gt_boxes, matched_flags


def _prepare_predictions(
    annotations_by_filename: Dict[str, List[dict]]
) -> List[dict]:
    """Flatten predictions into a list with tensors for downstream processing."""

    predictions: List[dict] = []
    for file_name, annotations in annotations_by_filename.items():
        for ann in annotations:
            bbox = ann.get("bbox")
            category_id = ann.get("category_id")
            if bbox is None or category_id is None:
                continue
            bbox_tensor = box_convert(
                torch.tensor(bbox, dtype=torch.float32), "xywh", "xyxy"
            )
            score = float(ann.get("score", 1.0))
            predictions.append(
                {
                    "file_name": file_name,
                    "category_id": category_id,
                    "bbox": bbox_tensor,
                    "score": score,
                }
            )
    return predictions


def compute_average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute the Average Precision (AP) given recall and precision curves."""

    if recalls.size == 0 or precisions.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def evaluate_detections(
    ground_truth_path: Path,
    prediction_path: Path,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """Evaluate detections and compute precision, recall, and mAP."""

    gt_data = read_coco_annotations(ground_truth_path)
    pred_data = read_coco_annotations(prediction_path)

    gt_annotations, _ = _group_annotations_by_filename(gt_data)
    pred_annotations, _ = _group_annotations_by_filename(pred_data)

    print(f"Number of ground truth images: {len(gt_annotations)}"
          f", Number of predicted images: {len(pred_annotations)}"
          )

    
    common_filenames = sorted(set(gt_annotations) & set(pred_annotations))
    if not common_filenames:
        return 0.0, 0.0, 0.0

    gt_boxes, matched_flags = _prepare_ground_truth(
        {name: gt_annotations[name] for name in common_filenames}
    )
    predictions = _prepare_predictions(
        {name: pred_annotations[name] for name in common_filenames}
    )

    # Sort predictions by score descending for proper precision/recall computation.
    predictions.sort(key=lambda item: item["score"], reverse=True)

    total_gt = sum(
        boxes.shape[0]
        for file_boxes in gt_boxes.values()
        for boxes in file_boxes.values()
    )
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    true_positives = []
    false_positives = []

    for pred in predictions:
        file_name = pred["file_name"]
        category_id = pred["category_id"]
        pred_box = pred["bbox"].unsqueeze(0)

        gt_file_boxes = gt_boxes.get(file_name, {})
        matched_file_flags = matched_flags.get(file_name, {})

        gt_category_boxes = gt_file_boxes.get(category_id)
        matched_category_flags = matched_file_flags.get(category_id)

        if gt_category_boxes is None or gt_category_boxes.numel() == 0:
            false_positives.append(1)
            true_positives.append(0)
            continue

        # Mask out already matched ground truth boxes.
        available_mask = ~matched_category_flags
        if not torch.any(available_mask):
            false_positives.append(1)
            true_positives.append(0)
            continue

        candidate_boxes = gt_category_boxes[available_mask]
        if candidate_boxes.numel() == 0:
            false_positives.append(1)
            true_positives.append(0)
            continue

        ious = ops.box_iou(pred_box, candidate_boxes).squeeze(0)
        max_iou, max_idx = torch.max(ious, dim=0)
        if float(max_iou) >= iou_threshold:
            true_positives.append(1)
            false_positives.append(0)
            # Update the matched flag for the selected ground truth box.
            global_index = torch.nonzero(available_mask, as_tuple=False)[max_idx]
            matched_category_flags[global_index] = True
        else:
            false_positives.append(1)
            true_positives.append(0)

    if not true_positives:
        return 0.0, 0.0, 0.0

    tp_cumulative = torch.cumsum(torch.tensor(true_positives, dtype=torch.float32), dim=0)
    fp_cumulative = torch.cumsum(torch.tensor(false_positives, dtype=torch.float32), dim=0)

    precisions = tp_cumulative / (tp_cumulative + fp_cumulative + 1e-8)
    recalls = tp_cumulative / total_gt

    ap = compute_average_precision(recalls.numpy(), precisions.numpy())
    precision = float(precisions[-1])
    recall = float(recalls[-1])

    return precision, recall, ap


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ground truth and predictions")
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to the ground truth COCO annotations (e.g. test_annotations.json)",
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="Path to the predicted COCO annotations (e.g. pred_annotations.json)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold to consider a detection a true positive (default: 0.5)",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    precision, recall, ap = evaluate_detections(
        args.ground_truth, args.predictions, args.iou_threshold
    )

    print(f"Evaluated {args.ground_truth.name} vs {args.predictions.name}")
    print(f"IoU threshold: {args.iou_threshold:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP: {ap:.4f}")


if __name__ == "__main__":
    main()
