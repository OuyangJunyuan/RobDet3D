import os
import json
import torch
import numpy as np

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# Segment Anything
from segment_anything import build_sam, SamPredictor


class Detector:
    """
    GroundingDINO
    """

    def __init__(self, grounded_checkpoint, text_prompt, box_threshold, text_threshold, device='cpu'):
        from groundingdino.config import GroundingDINO_SwinB

        config_file = GroundingDINO_SwinB.__file__
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = self.load_model(config_file, grounded_checkpoint, device=device)
        self.model.eval()
        self.model.to(device)
        self.device = device
        print(self.load_prompt())

    @staticmethod
    def load_model(model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        return model

    @staticmethod
    def load_image(image_path):
        from PIL import Image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def load_prompt(self):
        caption = self.text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        return caption

    @torch.no_grad()
    def forward(self, image, with_logits=True):
        caption = self.load_prompt()
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt.detach(), pred_phrases


class Segmentor:
    """
    Segment Anything
    """

    def __init__(self, sam_checkpoint, device='cpu'):
        self.predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        self.image = None
        self.device = device

    @property
    def image_shape(self):
        return self.image.shape[:2]

    def set_image(self, image_path):
        import cv2
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.predictor.set_image(image)

    @torch.no_grad()
    def forward(self, _boxes_filt):
        H, W = self.image_shape
        boxes_filt = _boxes_filt.detach().clone()
        if boxes_filt.size(0):
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                # xylw -> x1y1x2y2
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, self.image_shape).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.device),
                multimask_output=False,
            )
        else:
            masks = torch.zeros([0, 0, H, W])
        return masks.detach()


def save_visualization(output_dir, image, masks, boxes_filt, pred_phrases):
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_box(box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label)

    import matplotlib.pyplot as plt

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )
    plt.close()


def save_mask_data(output_dir, mask_list, box_list, label_list):
    import cv2

    value = 0  # 0 for background

    mask_img = np.zeros(mask_list.shape[-2:], dtype=np.uint8)
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    cv2.imwrite(os.path.join(output_dir, 'mask.png'), mask_img)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })

    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

    with open(os.path.join(output_dir, 'mask.json'), 'r') as f:
        saved_data = json.load(f)

    assert json_data == saved_data
