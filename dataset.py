import os
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset

class CompositionDataset(Dataset):
    """画像フォルダとラベルファイルを読み込むデータセットクラス。"""
    def __init__(self, image_dir: str, label_file: str, transform: Optional[callable] = None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line in label file: {line}")
                filename, label = parts
                label = int(label)
                full_path = os.path.join(self.image_dir, filename)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"{full_path} が見つかりません")
                self.samples.append((filename, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.image_dir, path)
        image = Image.open(full_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label
