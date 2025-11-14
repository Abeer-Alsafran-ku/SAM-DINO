# to compare the performance of ground truth masks and predicted masks
import os
import json
from pathlib import Path
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
import numpy as np


# read the coco annotations
def read_coco_annotations(coco_annotation_path):
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


