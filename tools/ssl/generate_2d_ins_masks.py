"""
Examples:
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

cd Grounded-Segment-Anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

Notes:
1. for better performance, you can download Gounding DINO Swin-B model from GoundingDINO official repo.https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
2. Its corresponding config file is stored in GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py.
3. It may take a few time to download BERT from huggingface hub at the first time you run this model.

References:
1. consuming 8.2G GPU MEMS.
2. takeing 1.15 second per image. (3087 imgs / 74 mins)
"""
import torch
from tqdm import tqdm
from pathlib import Path
from helper import grounded_sam

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "Car. Pedestrian. Cyclist."
IMAGE_DIR = "data/kitti_sparse/training/image_2"
MASK_DIR = "/home/nrsl/dataset/image_2_mask"

DEVICE = 'cuda'
GROUNDING_DINO_CKPT = "/home/nrsl/workspace/temp/Grounded-Segment-Anything/groundingdino_swinb_cogcoor.pth"
SEGMENT_ANTHING_CKPT = "/home/nrsl/workspace/codebase/segment-anything/models/sam_vit_h_4b8939.pth"

if __name__ == "__main__":

    output_dir = Path(MASK_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = grounded_sam.Detector(GROUNDING_DINO_CKPT, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD, DEVICE)
    segmentor = grounded_sam.Segmentor(SEGMENT_ANTHING_CKPT, DEVICE)

    file_list = list(Path(IMAGE_DIR).iterdir())
    file_list.sort()
    for i, image_path in enumerate(tqdm(iterable=file_list)):
        frame_dir = output_dir / image_path.stem
        if frame_dir.exists():
            continue
        frame_dir.mkdir()
        image_pil, image = detector.load_image(image_path.__str__())
        boxes_filt, pred_phrases = detector.forward(image)
        segmentor.set_image(image_path.__str__())
        masks = segmentor.forward(boxes_filt)

        grounded_sam.save_mask_data(frame_dir, masks, boxes_filt, pred_phrases)
        # if i < 1000:
        #     grounded_sam.save_visualization(frame_dir, segmentor.image, masks, boxes_filt, pred_phrases)
