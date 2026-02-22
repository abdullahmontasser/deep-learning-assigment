# Smart Botanical Flower Identifier

Fine-tuning ResNet18 on the Oxford Flowers-102 dataset to classify 102 flower species.

## How to run

```bash
pip install torch torchvision
python main.py
```

The dataset downloads automatically on first run (~350 MB).

## Approach

Pre-trained ResNet18 (ImageNet) modified with a custom classification head (512 → 256 → 102).

Fine-tuning is done in two stages:
1. **Stage 1** — backbone frozen, only the new head is trained
2. **Stage 2** — last residual block (`layer4`) unfrozen and fine-tuned with a smaller learning rate

## Results

| Metric | Score |
|---|---|
| Top-1 Accuracy | 64.04% |
| Top-5 Accuracy | 87.35% |
