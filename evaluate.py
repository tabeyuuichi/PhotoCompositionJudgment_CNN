import argparse
import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import CompositionDataset
from model import SimpleCNN


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0


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

    acc = evaluate(model, dataloader, device)
    print(f"Accuracy: {acc * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate composition classifier')
    parser.add_argument('image_dir', help='Directory with evaluation images')
    parser.add_argument('label_file', help='Text file with labels for evaluation images')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of composition classes')
    parser.add_argument('--batch-size', type=int, default=32)
    main(parser.parse_args())
