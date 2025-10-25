# Grounded-SAM Fine-Tuning Helper

This repository provides a light-weight training script for fine-tuning the
[Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
stack on your own dataset. The entry point is [`main.py`](main.py) which wraps
Grounding DINO for detection training and (optionally) Segment Anything for
mask supervision.

## Installation

The script expects the official GroundingDINO and Segment Anything packages to
be available in your environment. Install them from source so that both modules
are on the `PYTHONPATH`:

```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install pycocotools
```

## Dataset format

Training data must follow the COCO format:

* `images`: list of image metadata objects (each containing at least `id` and
  `file_name`).
* `annotations`: list of annotations referencing the `image_id`. Each
  annotation must contain:
  * `bbox`: `[x, y, width, height]` in absolute pixel coordinates.
  * `category_id`: integer label.
  * `segmentation`: polygon or RLE mask information if you plan to supervise
    SAM (required when `--return-masks` is set). A fallback `mask_path` field
    pointing to a binary mask file inside the image directory is also supported.
* `categories`: (recommended) list of `{ "id": int, "name": str }` entries to
  define the label set. When omitted, the script will infer the label ids from
  the annotations.

Images referenced in the JSON must be located inside the directory passed to
`--images`.

## Usage

```bash
python main.py \
  --annotations /path/to/annotations.json \
  --images /path/to/images_dir \
  --config /path/to/groundingdino/config.py \
  --checkpoint /path/to/groundingdino_checkpoint.pth \
  --sam-checkpoint /path/to/sam_checkpoint.pth \
  --sam-backbone vit_h \
  --return-masks \
  --finetune-sam \
  --epochs 50 \
  --batch-size 4 \
  --lr 1e-4 \
  --image-size 1024
```

### Important flags

* `--return-masks`: loads segmentation masks from the dataset and enables SAM
  supervision. Required if you plan to fine-tune SAM via `--finetune-sam`.
* `--finetune-sam`: unfreezes the specified SAM backbone and optimises it using
  a combined binary cross-entropy + Dice loss on the provided masks.
* `--sam-backbone`: selects the SAM backbone registered in `segment_anything`.
* `--sam-model`: deprecated compatibility flag. When provided with a path it is
  treated like `--sam-checkpoint`; otherwise it maps to `--sam-backbone`.
* `--image-size`: the resolution to which each image (and its annotations) will
  be resized before being fed to Grounding DINO and SAM.

Checkpoints are written to `--output` after every epoch with both the model and
optimizer state. The script uses mixed precision automatically when CUDA is
available.

## Notes

* The script assumes a single GPU. For multi-GPU training, wrap the model in
  `DistributedDataParallel` and switch the data loader to a distributed sampler.
* Ensure that the Grounding DINO config's `NUM_CLASSES` matches the number of
  categories in your dataset; otherwise training will raise an error.
* When masks are provided, they can be encoded as polygons, compressed RLE, or
  external grayscale images referenced via `mask_path`.
