# 写真構図チェッカー

このリポジトリは、画像フォルダとラベルファイルから簡単な CNN を学習し、写真の構図を分類するためのサンプル実装です。

## 必要な環境

Python 3.8 以上と以下のパッケージを使用します。

```bash
pip install torch torchvision pillow
```

## データセット形式

1. **画像フォルダ**: 学習に使用する画像をすべて格納したディレクトリ。
2. **ラベルファイル**: 各行にラベル番号のみを記述したテキストファイル。画像フォルダにあるファイル名順 (アルファベット順) にラベルを並べます。

```
0
1
0
```

ラベル番号は画像の正しい構図クラスを表します。画像数とラベル数が一致する必要があります。

## 学習方法

以下のコマンドで学習を実行し、`--save-model` で指定したパスに学習済みモデルを保存します。

```bash
python train.py /path/to/train_images train_labels.txt --num-classes 3 --epochs 20 --save-model model.pth
```

## 評価方法

保存したモデルを使用して別のデータセットを評価する場合は `evaluate.py` を実行します。

```bash
python evaluate.py /path/to/test_images test_labels.txt --model-path model.pth --num-classes 3
```

これにより正解率が表示されます。学習用と評価用のデータセットを分けることでモデルの性能を確認できます。

## モデルファイルについて

`train.py` の `--save-model` オプションで指定したファイルに学習済みモデルが保存されます。評価時にはそのファイルを `--model-path` に渡してください。

以上で、データセットの準備から学習・評価まで一通り実行できます。
PyTorch と torchvision をインストールした上で次のコマンドを実行します。
