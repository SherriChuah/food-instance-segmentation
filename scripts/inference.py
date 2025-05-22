import os
import torch
import torchvision

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load pretrained model
model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_prediction(img_path: str, threshold: float=0.5):
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

        pred_score = prediction[0]['scores'].cpu().numpy()
        pred_labels = prediction[0]['labels'].cpu().numpy()
        pred_boxes = prediction[0]['boxes'].cpu().numpy()
        pred_masks = prediction[0]['masks'].cpu().numpy()

        selected = [i for i, score in enumerate(pred_score) if score > threshold]

        boxes = pred_boxes[selected]
        labels = pred_labels[selected]
        masks = pred_masks[selected]

        return img, boxes, labels, masks
    

def draw_predictions(img, boxes, labels, masks, output_path):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i in range(len(boxes)):
        box = boxes[i]
        label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
        draw.rectangle(box.tolist(), outline="red", width=3)
        draw.text((box[0], box[1]), label, fill="red", font=font)


        # add mask overlay
        mask = masks[i,0]
        mask_img = Image.fromarray((mask * 255).astype('uint8')).resize(img.size)
        img.paste(Image.new("RGBA", img.size, (255, 0, 0, 100)), mask=mask_img)
    
    img.save(output_path)


if __name__ == "__main__":
    input_dir = 'images/'
    output_dir = 'outputs/'
    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(input_dir):
        print(img_file)
        if img_file.endswith((".jpg", ".png")):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"pred_{img_file}")

            img, boxes, labels, masks = get_prediction(img_path, threshold=0.7)
            draw_predictions(img, boxes, labels, masks, output_path)

            print(f"Saved prediction to {output_path}")