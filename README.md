# Mechatronics Sentence Embeddings Pipeline

End-to-end machine learning pipeline to train domain-specific sentence embeddings for mechatronics from scratch using low-data techniques.

## Overview

This pipeline:
1. **Scrapes** mechatronics-related text from Wikipedia, arXiv, and technical documentation
2. **Cleans & balances** the dataset across 8 topic buckets
3. **Trains a tokenizer** (WordPiece) from scratch on domain text
4. **Pretrains a tiny BERT** (4 layers, 256 hidden) with Masked Language Modeling
5. **Trains sentence embeddings** using TSDAE (denoising autoencoder) + SimCSE (contrastive learning)
6. **Evaluates** on a test set and visualizes embeddings in 3D with PCA

## Low-Data Techniques

- **TSDAE**: Transformer-based Sequential Denoising Auto-Encoder for unsupervised sentence embedding learning
- **SimCSE**: Simple Contrastive Learning using dropout as noise for positive pairs
- **In-batch negatives**: Efficient contrastive learning with MultipleNegativesRankingLoss
- **Tiny architecture**: 4-layer, 256-hidden BERT (~3.5M parameters) for fast training
- **Domain tokenizer**: Custom WordPiece vocabulary (16k tokens) specialized for mechatronics

## Requirements

- Python 3.8+
- CUDA GPU (optional but recommended)
- ~5GB disk space for data
- ~2GB RAM minimum

## Installation

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline

```powershell
.\scripts\run_pipeline.ps1
```

### Run Individual Steps

```powershell
# 1. Scrape data
python src/scrape/crawl.py --config configs/pipeline.yaml
python src/scrape/extract.py --config configs/pipeline.yaml

# 2. Clean and balance
python src/data/analyze.py --config configs/pipeline.yaml
python src/data/clean.py --config configs/pipeline.yaml
python src/data/balance.py --config configs/pipeline.yaml

# 3. Train tokenizer
python src/tokenizer/train_tokenizer.py --config configs/pipeline.yaml

# 4. Pretrain encoder with MLM
python src/train/pretrain_mlm.py --config configs/pipeline.yaml

# 5. Train TSDAE
python src/train/train_tsdae.py --config configs/pipeline.yaml

# 6. Train SimCSE
python src/train/train_simcse.py --config configs/pipeline.yaml

# 7. Evaluate and visualize
python src/eval/prepare_testset.py --config configs/pipeline.yaml
python src/eval/visualize_pca.py --config configs/pipeline.yaml
```

### Skip Certain Steps

```powershell
# Skip scraping (use existing data)
.\scripts\run_pipeline.ps1 -SkipScrape

# Skip MLM pretraining
.\scripts\run_pipeline.ps1 -SkipMLM

# Skip all training, just evaluate
.\scripts\run_pipeline.ps1 -SkipMLM -SkipTSDAE -SkipSimCSE
```

## Configuration

Edit `configs/pipeline.yaml` to customize:
- Scraping sources and limits
- Topic keywords and balancing
- Model architecture (hidden size, layers, heads)
- Training hyperparameters (batch size, learning rate, epochs)
- Evaluation settings

## Topic Buckets

The pipeline categorizes text into 8 mechatronics topics:
- **Control**: PID controllers, feedback systems, servo control
- **Robotics**: Manipulators, kinematics, path planning
- **Sensors**: Encoders, LiDAR, IMU, transducers
- **Actuators**: Motors, hydraulics, pneumatics
- **PLC**: Programmable logic controllers, SCADA, HMI
- **Embedded**: Microcontrollers, firmware, RTOS
- **Kinematics**: Forward/inverse kinematics, Jacobians
- **Dynamics**: Forces, torques, equations of motion

## Directory Structure

```
.
├── configs/
│   └── pipeline.yaml          # Master configuration
├── data/
│   ├── raw/                   # Scraped HTML and JSON
│   ├── interim/               # Extracted text
│   ├── clean/                 # Cleaned data
│   ├── balanced/              # Balanced corpus
│   ├── tokenizer/             # Trained tokenizer
│   └── splits/                # Train/val/test splits
├── artifacts/
│   ├── models/
│   │   ├── tiny-bert-mlm/     # MLM pretrained encoder
│   │   ├── tsdae-embeddings/  # TSDAE model
│   │   └── mecha-embed-v1/    # Final SimCSE model
│   ├── logs/                  # Analysis reports and plots
│   └── eval/                  # Test embeddings and visualization
├── src/
│   ├── scrape/                # Web crawling and extraction
│   ├── data/                  # Data processing
│   ├── tokenizer/             # Tokenizer training
│   ├── models/                # Model architectures
│   ├── train/                 # Training scripts
│   └── eval/                  # Evaluation and visualization
└── scripts/
    └── run_pipeline.ps1       # Pipeline orchestration
```

## Outputs

- **Final model**: `artifacts/models/mecha-embed-v1/best/`
- **Tokenizer**: `data/tokenizer/tokenizer.json`
- **Test embeddings**: `artifacts/eval/test_embeddings.npz`
- **3D visualization**: `artifacts/eval/pca_3d.html` (open in browser)
- **Analysis plots**: `artifacts/logs/analysis_plots.png`

## Using the Trained Model

```python
import torch
from transformers import BertModel
from tokenizers import Tokenizer
import numpy as np

# Load model and tokenizer
model = BertModel.from_pretrained('artifacts/models/mecha-embed-v1/best/')
tokenizer = Tokenizer.from_file('data/tokenizer/tokenizer.json')

def encode(text, model, tokenizer, device='cpu'):
    """Encode text to embedding."""
    model.eval()
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids]).to(device)
    attention_mask = torch.tensor([[1] * len(encoding.ids)]).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # L2 normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    
    return embeddings.cpu().numpy()

# Example
text = "The PID controller regulates motor speed using feedback."
embedding = encode(text, model, tokenizer)
print(f"Embedding shape: {embedding.shape}")
```

## References

- **TSDAE**: Wang et al., "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning" (EMNLP 2021)
- **SimCSE**: Gao et al., "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (EMNLP 2021)
- **Sentence-BERT**: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (EMNLP 2019)

## License

This project is for educational and research purposes.

