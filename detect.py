import torch
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint_path = '/content/a-PyTorch-Tutorial-to-Object-Detection/checkpoint_ssd300.pth.tar'

from model import SSD300
torch.serialization.add_safe_globals([SSD300])

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect(original_image, min_score=0.2, max_overlap=0.45, top_k=200, suppress=None):
    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to device
    image = image.to(device)

    # Forward prop
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects
    det_boxes, det_labels, det_scores = model.detect_objects(
        predicted_locs, predicted_scores,
        min_score=min_score,
        max_overlap=max_overlap,
        top_k=top_k
    )

    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor([
        original_image.width, original_image.height,
        original_image.width, original_image.height
    ]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    if det_labels == ['background']:
        return original_image

    # Annotate
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("./calibril.ttf", 15)
    except:
        font = ImageFont.load_default()

    for i in range(det_boxes.size(0)):
        if suppress is not None and det_labels[i] in suppress:
            continue

        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])

        text_size = font.getbbox(det_labels[i].upper())
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        text_location = [box_location[0] + 2., box_location[1] - text_height]
        textbox_location = [
            box_location[0],
            box_location[1] - text_height,
            box_location[0] + text_width + 4.,
            box_location[1]
        ]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)

    del draw
    return annotated_image

if __name__ == '__main__':
    img_path = '/content/BCCD_Dataset/BCCD/JPEGImages/BloodImage_00002.jpg'
    original_image = Image.open(img_path).convert('RGB')
    annotated = detect(original_image, min_score=0.2, max_overlap=0.45, top_k=200)
    plt.imshow(annotated)
    plt.axis('off')
    plt.show()
