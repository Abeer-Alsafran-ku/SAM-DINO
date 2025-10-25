import os
import json
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
from torchvision.ops import box_iou
import torchvision.transforms.functional as F
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from segment_anything import sam_model_registry,SamPredictor
from groundingdino.util.slconfig import SLConfig
from groundingdino.models.criterion import SetCriterion

# from groundingdino.models.criterion import SetCriterion
# from groundingdino.groundingdino import SetCriterion
# from groundingdino.criterion import SetCriterion
# from groundingdino.models.GroundingDINO import SetC /riterion

config_path = "/content/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
groundingdino_config = SLConfig.fromfile(config_path)
groundingdino_config.device = "cuda"

train_labels_path = '/content/KDrone-4/train/_annotations.coco.json'
dataset_dir = '/content/drive/MyDrive/kvec'
train_images_path = '/content/KDrone-4/train'
classes_file = '/content/drive/MyDrive/kvec/classes.txt'
model_checkpoint_path = "/content/drive/MyDrive/models/groundingdino_swint_ogc.pth"
sam_checkpoint = "/content/drive/MyDrive/models/sam_vit_h_4b8939.pth"

def build_grounding_dino_model(checkpoint_path=None, device='cuda'):
    """Load a Grounding DINO model. Replace with the actual import path from the repo you cloned.

    This function expects that the repo exposes a function/class to build the model. If not, adapt accordingly.
    """
    try:
        # Example placeholder import. In the original Grounded-SAM / GroundingDINO codebase the model
        # is created via a config and model builder. Replace these lines to match the exact API.
        # from groundingdino.models import build_model
        # from groundingdino.util.misc import get_state_dict
        # The repo normally uses a config to instantiate. If you can't import build_model, adapt.
        model = build_model()
        if checkpoint_path is not None:
            sd = torch.load(checkpoint_path, map_location='cpu')
            state_dict =  clean_state_dict(sd)
            model.load_state_dict(state_dict, strict=False)
        model.to(device)
        return model
    except Exception as e:
        print('Could not import groundingdino builder. You must update build_grounding_dino_model() to match the repo API.', e)
        raise


def freeze_sam_except_decoder(sam_model, unfreeze_decoder=True):
    # freeze everything except the mask decoder (or selective layers) to save VRAM
    for name, p in sam_model.named_parameters():
        p.requires_grad = False
    if unfreeze_decoder:
        for name, p in sam_model.mask_decoder.named_parameters():
            p.requires_grad = True

def build_grounding_dino_model_and_criterion(config, checkpoint_path=None, device='cuda'):
    """Builds Grounding-DINO model and SetCriterion (Hungarian matcher + losses)"""

    model = build_model(config)
    if checkpoint_path is not None:
        sd = torch.load(checkpoint_path, map_location='cpu')
        state_dict = clean_state_dict(sd)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Hungarian matcher with default weights from config
    matcher = build_matcher(config)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = SetCriterion(num_classes=config.MODEL.DINO.NUM_CLASSES, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1)
    criterion.to(device)

    return model, criterion
class CocoLikeDataset(Dataset):
    def __init__(self, images_dir, coco_json_path, transforms=None, return_masks=False):
        with open(coco_json_path, 'r') as f:
            self.coco = json.load(f)
        self.images_dir = images_dir
        self.id2img = {img['id']: img for img in self.coco['images']}
        self.imgid_to_anns = {}
        for ann in self.coco['annotations']:
            self.imgid_to_anns.setdefault(ann['image_id'], []).append(ann)
        self.image_ids = list(self.id2img.keys())
        self.transforms = transforms
        self.return_masks = return_masks

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_meta = self.id2img[img_id]
        img_path = os.path.join(self.images_dir, img_meta['file_name'])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        anns = self.imgid_to_anns.get(img_id, [])
        boxes = []
        labels = []
        masks = []
        for a in anns:
            x, y, bw, bh = a['bbox']
            xmin = x
            ymin = y
            xmax = x + bw
            ymax = y + bh
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(a['category_id'])
            if self.return_masks and 'mask_path' in a:
                mask = np.array(Image.open(a['mask_path']).convert('L')) > 128
                masks.append(mask.astype(np.uint8))
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros((0,), dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
        if self.return_masks:
            if masks:
                masks = torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8)
            else:
                masks = torch.zeros((0, h, w), dtype=torch.uint8)
            target['masks'] = masks

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)
        return img, target
def train(
    coco_train_json,
    images_dir,
    groundingdino_config,
    groundingdino_ckpt,
    sam_ckpt=None,
    out_dir='/content/drive/MyDrive/gsam_finetune',
    epochs=100,
    batch_size=32,
    lr=1e-4,
    device='cuda',
    finetune_sam=False,
    return_masks=False
):

    os.makedirs(out_dir, exist_ok=True)
    print("~~~~~~~~~ Started Training ~~~~~~~~~")
    # dataset & dataloader
    transforms = T.Compose([T.Resize((800, 800)), T.ToTensor()])
    train_dataset = CocoLikeDataset(images_dir, coco_train_json, transforms=transforms, return_masks=return_masks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, criterion = build_grounding_dino_model_and_criterion(groundingdino_config, groundingdino_ckpt, device=device)

    sam_model = None
    if sam_ckpt is not None:
        sam_model = sam_model_registry['vit_h'](checkpoint=sam_ckpt).to(device)
        if not finetune_sam:
            for p in sam_model.parameters():
                p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    if sam_model is not None:
        params += [p for p in sam_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs = torch.stack(imgs).to(device)
            # Targets need to be a list of dicts with 'boxes', 'labels', 'masks' (if any)
            processed_targets = []
            for t in targets:
                target = {k: v.to(device) for k, v in t.items() if k in ['boxes','labels']}
                if return_masks and 'masks' in t:
                    target['masks'] = t['masks'].to(device)
                processed_targets.append(target)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                # Compute proper Grounding-DINO losses using SetCriterion
                loss_dict = criterion(outputs, processed_targets)
                loss = sum(loss_dict[k] for k in loss_dict.keys())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}')

        ckpt_path = os.path.join(out_dir, f'finetuned_epoch{epoch+1}.pth')
        torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)
        print('Saved checkpoint:', ckpt_path)

    print('Training complete. Checkpoints saved in', out_dir)
train(train_labels_path, train_images_path, groundingdino_config=groundingdino_config, groundingdino_ckpt=model_checkpoint_path, sam_ckpt=sam_checkpoint, epochs=100, batch_size=32, lr=1e-5, finetune_sam=False, return_masks=False)

I have the above code and i want to train grounding sam on my dataset, i am facing issues regarding compatibility with google colab, some grounding dino imports are not being compiled properly. i have installed sam and grounding dino.  
