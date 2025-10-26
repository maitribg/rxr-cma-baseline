# RxR CMA Baseline Evaluation

Minimal implementation for evaluating the Cross-Modal Attention (CMA) baseline on the RxR-Habitat dataset.

## Quick Start (Google Colab)

```python
# Install dependencies
!pip install -q habitat-sim==0.3.1 habitat-lab==0.3.1 torch torchvision tqdm

# Clone repo
!git clone https://github.com/yourusername/rxr-cma-baseline-eval.git
%cd rxr-cma-baseline-eval

# Mount Google Drive (with your data)
from google.colab import drive
drive.mount('/content/drive')

# Run evaluation
!python eval.py \
  --checkpoint /content/drive/MyDrive/rxr_data/checkpoints/ckpt.0.pth \
  --dataset /content/drive/MyDrive/rxr_data/datasets/RxR_VLNCE_v0 \
  --scenes /content/drive/MyDrive/rxr_data/scene_datasets/mp3d \
  --text-features /content/drive/MyDrive/rxr_data/datasets/RxR_VLNCE_v0/text_features \
  --split val_unseen \
  --num-episodes 10
```

## Data Requirements

- RxR dataset episodes (`.json.gz` files)
- Matterport3D scenes
- BERT text features
- Pretrained CMA checkpoint

## Citation

Based on the VLN-CE paper: https://arxiv.org/abs/2004.02857