import torch
from datasets import PascalVOCDataset
from utils import *
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Параметры
data_folder = '/content/BCCD_Dataset'
keep_difficult = True 
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = './checkpoint_ssd300.pth.tar'


from model import SSD300, VGGBase

torch.serialization.add_safe_globals([SSD300, VGGBase])


checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model = checkpoint['model'].to(device)
model.eval()


test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Оценка модели на тестовом наборе.

    :param test_loader: DataLoader для тестовых данных
    :param model: модель
    """

    model.eval()

    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)

            # Forward
            predicted_locs, predicted_scores = model(images)

            # Детекция объектов
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores,
                min_score=0.01, max_overlap=0.45, top_k=200
            )

            # Перенос GT на device
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

 
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores,
                                  true_boxes, true_labels, true_difficulties)

    # аP для каждого класса
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
