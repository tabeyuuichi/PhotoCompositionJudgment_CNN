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
        image_files = [f for f in os.listdir(self.image_dir)
                       if os.path.isfile(os.path.join(self.image_dir, f))]
        image_files.sort()
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != len(image_files):
            raise ValueError("画像の枚数とラベルの数が一致しません")
        for filename, label_str in zip(image_files, labels):
            full_path = os.path.join(self.image_dir, filename)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"{full_path} が見つかりません")
            label = int(label_str)
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
