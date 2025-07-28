import argparse
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, fbeta_score

from dataset import CompositionDataset
from model import SimpleCNN


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    f05 = fbeta_score(all_labels, all_preds, beta=0.5, average="macro", zero_division=0)
    return accuracy, precision, recall, f1, f05


def main(args):
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
    dataset = CompositionDataset(args.image_dir, args.label_file, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    acc, prec, recall, f1, f05 = evaluate(model, dataloader, device)
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"F0.5-score: {f05:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate composition classifier')
    parser.add_argument('image_dir', help='Directory with evaluation images')
    parser.add_argument('label_file', help='Text file with labels for evaluation images')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of composition classes')
    parser.add_argument('--batch-size', type=int, default=32)
    main(parser.parse_args())
