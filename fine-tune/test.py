from groundingdino.util.inference import load_model, load_image, predict, annotate
# import cv2
import json
from pathlib import Path
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
import os

def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.3):
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    # print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))

    if not nms_boxes_list:
        return boxes, logits, phrases

    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list


def create_coco_annotations(boxes, logits, phrases, image_source, image_path, image_id ):
    
    image_height, image_width = image_source.shape[:2]
    
    images = [{
        "id": image_id,
        "file_name":image_path,
        "width": image_width,
        "height": image_height,
    }]

    category_mapping = {
        'bus' : 1,
        'car' : 2,
        'truck' : 3,
        'pickup-truck' : 4,
        'van' : 5
    }
    annotations = []

    annotation_id = 1

    if boxes.is_cuda:
        boxes = boxes.cpu()
    if logits.is_cuda:
        logits = logits.cpu()

    for box, logit, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
        category_id = category_mapping[phrase]

        x_center, y_center, width_rel, height_rel = box
        x = (x_center - width_rel / 2) * image_width
        y = (y_center - height_rel / 2) * image_height
        width = width_rel * image_width
        height = height_rel * image_height

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, width, height],
            "area": width * height,
            "score": float(logit),
            "iscrowd": 0,
        })
        annotation_id += 1
    

    return {
        "images": images,
        "annotations": annotations,
    }


def process_image(
        model_config="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights="weights/model_weights0_12.5.pth",
        image_path="/home/abeer/roboflow/test",
        text_prompt="bus . car . truck . pickup-truck . van",
        output_json_path="output_annotations.json",
        image_name=None,
        image_id=0,
        box_threshold=0.25,
        text_threshold=0.2

):
    model = load_model(model_config, model_weights)
    #model.load_state_dict(torch.load(state_dict_path))
    image_source, image = load_image(os.path.join(image_path, image_name))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
    for phrase in phrases:
        phrase_new  = phrase.split(' ')[0]
        if phrase_new == 'pickup':
            phrase_new = 'pickup-truck'
        phrases[phrases.index(phrase)] = phrase_new
    
    coco_output = create_coco_annotations(boxes, logits, phrases, image_source, image_name,image_id)

    with open(output_json_path, "a", encoding="utf-8") as json_file:
        json_file.extend(coco_output['images'])
        json_file.extend(coco_output['annotations'])



if __name__ == "__main__":
    model_weights="weights/groundingdino_swint_ogc.pth"
    model_weights="weights/model_weights0_12.5.pth"
    # img_path = "/home/abeer/roboflow/train/ministrey_zone_2_Flight_01_01941_JPG.rf.f6dd851625ce27434b3ae9087aed4767.jpg"
    prompt = "bus . car . truck . pickup-truck . van"
    # loop over all images in a directory
    image_path = "/home/abeer/roboflow/test"
    i = 0 
    for image_name in os.listdir(image_path):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Processing image: {os.path.join(image_path, image_name)}")
            process_image(model_weights=model_weights,text_prompt = prompt,image_name=image_name,image_id=i)
            i += 1
    
