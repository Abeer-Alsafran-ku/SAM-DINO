import os
import glob
import numpy as np
from sklearn.metrics import average_precision_score

def read_yolo_labels(file_path):
    boxes = []
    if not os.path.exists(file_path):
        return boxes
    with open(file_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id, x, y, w, h = parts
            # Convert to (x1, y1, x2, y2)
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append([int(class_id), x1, y1, x2, y2]) # Ensure class_id is int
    return boxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def match_predictions(gt_boxes, pred_boxes, iou_threshold=0.5):
    matches = []
    used_preds = set()
    gt_matched = [False] * len(gt_boxes) # To keep track of matched ground truth boxes
    for i, gt in enumerate(gt_boxes):
        gt_class = gt[0]
        gt_coords = gt[1:]
        best_iou = 0
        best_idx = -1
        for j, pred in enumerate(pred_boxes):
            if j in used_preds:
                continue
            pred_class = pred[0]
            pred_coords = pred[1:]
            if gt_class != pred_class:
                continue
            iou = compute_iou(gt_coords, pred_coords)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_threshold and best_idx != -1:
            matches.append((gt, pred_boxes[best_idx], best_iou))
            used_preds.add(best_idx)
            gt_matched[i] = True # Mark ground truth as matched

    tp = len(matches)
    fn = len([m for m in gt_matched if not m]) # Unmatched ground truth are false negatives
    fp = len(pred_boxes) - tp # Predictions not matched are false positives

    return tp, fp, fn

def calculate_metrics(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Function to calculate mAP (simplified for this example, typically requires more sophisticated handling of confidence scores)
def calculate_map(gt_boxes_list, pred_boxes_list, iou_threshold=0.5):
    all_detections = []
    all_ground_truths = []
    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
      for pred in pred_boxes:
          all_detections.append({
              'bbox': pred[1:],
              'score': 0.35,
              'class_id': pred[0],
              'file_name': '' # Placeholder for file name if needed
          })
      for gt in gt_boxes:
           all_ground_truths.append({
              'bbox': gt[1:],
              'class_id': gt[0],
              'file_name': '', # Placeholder for file name if needed
              'matched': False # To track if a ground truth is matched
          })

    # This is a very simplified mAP calculation. A proper mAP calculation requires sorting detections by confidence and iterating through different recall levels.
    # For a more accurate mAP, consider using a dedicated library like pycocotools or implementing a more detailed calculation.
    # Here, we'll just demonstrate a basic approach based on matched predictions at a fixed IoU threshold.

    true_positives = []
    scores = []
    gt_classes = []
    pred_classes = []

    for det in all_detections:
      best_iou = 0.5
      best_gt_idx = -1
      for i, gt in enumerate(all_ground_truths):
        if gt['class_id'] == det['class_id'] and not gt['matched']:
          iou = compute_iou(det['bbox'], gt['bbox'])
          if iou > best_iou:
            best_iou = iou
            best_gt_idx = i
      if best_iou >= iou_threshold and best_gt_idx != -1:
        true_positives.append(1)
        all_ground_truths[best_gt_idx]['matched'] = True
      else:
        true_positives.append(0)
      scores.append(det['score'])
      gt_classes.append(det['class_id']) # This is not entirely correct for mAP, needs pairing with matched GT
      pred_classes.append(det['class_id']) # This is not entirely correct for mAP


    # Simplified AP calculation per class
    class_ids = sorted(list(set(gt_classes + pred_classes)))
    average_precisions = []

    for class_id in class_ids:
      class_tps = [true_positives[i] for i in range(len(all_detections)) if pred_classes[i] == class_id]
      class_scores = [scores[i] for i in range(len(all_detections)) if pred_classes[i] == class_id]
      class_gt_count = len([gt for gt in all_ground_truths if gt['class_id'] == class_id])

      if class_gt_count == 0:
          continue

      # Sort detections by score
      sorted_indices = np.argsort(class_scores)[::-1]
      sorted_tps = [class_tps[i] for i in sorted_indices]

      # Calculate precision and recall at each detection
      tp_cumsum = np.cumsum(sorted_tps)
      fp_cumsum = np.cumsum([1 - tp for tp in sorted_tps])

      precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
      recalls = tp_cumsum / class_gt_count

      # Append sentinel values to the ends
      precisions = np.concatenate(([1.], precisions, [0.]))
      recalls = np.concatenate(([0.], recalls, [1.]))

      # Compute AP using the 11-point interpolation method (or all-point interpolation in modern COCO)
      # Here we use a simplified approach
      ap = 0
      for t in np.arange(0., 1.1, 0.1):
          if np.sum(recalls >= t) == 0:
              p = 0
          else:
              p = np.max(precisions[recalls >= t])
          ap = ap + p / 11.
      average_precisions.append(ap)

    mAP = np.mean(average_precisions) if average_precisions else 0
    return mAP



# Example usage | Method 1
gt_path = '/home/abeer/dataset/test/labels'
# pred_path = '/home/abeer/test_swint_yolo' # No training on this path, just predictions
# pred_path = '/home/abeer/pred2yolo' # With training on this path
pred_path = '/home/abeer/pred' # With training on this path

print(f"GT Path: {gt_path}\nPred Path: {pred_path}")

all_files = glob.glob(os.path.join(gt_path, "*.txt"))

total_tp = 0
total_fp = 0
total_fn = 0

gt_boxes_list = []
pred_boxes_list = []

for file in all_files:
    filename = os.path.basename(file)
    gt_boxes = read_yolo_labels(os.path.join(gt_path, filename))
    pred_boxes = read_yolo_labels(os.path.join(pred_path, filename))

    gt_boxes_list.append(gt_boxes)
    pred_boxes_list.append(pred_boxes)

    tp, fp, fn = match_predictions(gt_boxes, pred_boxes)
    total_tp += tp
    total_fp += fp
    total_fn += fn

precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_fn)
mAP = calculate_map(gt_boxes_list, pred_boxes_list)


print(f"Total True Positives (TP): {total_tp}")
print(f"Total False Positives (FP): {total_fp}")
print(f"Total False Negatives (FN): {total_fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Average Precision (mAP): {mAP:.4f}")