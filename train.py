import argparse
import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import CompositionDataset
from model import SimpleCNN


def main(args):
    transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
    dataset = CompositionDataset(args.image_dir, args.label_file, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleCNN(num_classes=args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train composition classifier')
    parser.add_argument('image_dir', help='Directory with training images')
    parser.add_argument('label_file', help='Text file with image labels')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of composition classes')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save-model', help='Path to save trained model')
    main(parser.parse_args())
