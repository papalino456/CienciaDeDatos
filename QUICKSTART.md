# Quick Start Guide

## Prerequisites

- Windows 10/11 with PowerShell
- Python 3.8 or higher
- ~5GB free disk space
- (Optional) CUDA-capable GPU for faster training

## Installation (5 minutes)

1. **Clone or navigate to the project directory**
```powershell
cd CienciaDeDato
```

2. **Run setup script**
```powershell
.\scripts\setup.ps1
```

This will:
- Create a Python virtual environment (`.venv`)
- Install all required dependencies
- Verify your Python installation

## Running the Pipeline

### Option 1: Full Pipeline (4-8 hours)

Run everything from scraping to visualization:

```powershell
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Run full pipeline
.\scripts\run_pipeline.ps1
```

### Option 2: Step-by-Step Execution

Run individual stages:

```powershell
# Activate environment first
.\.venv\Scripts\Activate.ps1

# Stage 1: Data Collection (30-60 min)
python src/scrape/crawl.py --config configs/pipeline.yaml
python src/scrape/extract.py --config configs/pipeline.yaml

# Stage 2: Data Processing (10-20 min)
python src/data/analyze.py --config configs/pipeline.yaml
python src/data/clean.py --config configs/pipeline.yaml
python src/data/balance.py --config configs/pipeline.yaml

# Stage 3: Tokenizer (5-10 min)
python src/tokenizer/train_tokenizer.py --config configs/pipeline.yaml

# Stage 4: Model Training (2-4 hours)
python src/train/pretrain_mlm.py --config configs/pipeline.yaml
python src/train/train_tsdae.py --config configs/pipeline.yaml
python src/train/train_simcse.py --config configs/pipeline.yaml

# Stage 5: Evaluation (5-10 min)
python src/eval/prepare_testset.py --config configs/pipeline.yaml
python src/eval/visualize_pca.py --config configs/pipeline.yaml
```

### Option 3: Skip Stages

Skip time-consuming stages if you already have intermediate results:

```powershell
# Skip scraping (use existing data)
.\scripts\run_pipeline.ps1 -SkipScrape

# Skip everything up to training
.\scripts\run_pipeline.ps1 -SkipScrape -SkipClean -SkipTokenizer

# Only run evaluation (requires trained model)
.\scripts\run_pipeline.ps1 -SkipScrape -SkipClean -SkipTokenizer -SkipMLM -SkipTSDAE -SkipSimCSE
```

## Expected Outputs

After completion, you'll find:

1. **Final Model**: `artifacts/models/mecha-embed-v1/best/`
   - Use this for encoding mechatronics sentences

2. **3D Visualization**: `artifacts/eval/pca_3d.html`
   - Open in browser to explore sentence embeddings
   - Interactive plot with topic coloring

3. **Logs & Reports**:
   - `artifacts/logs/analysis_report.json` - Dataset statistics
   - `artifacts/logs/cleaning_stats.json` - Cleaning metrics
   - `artifacts/logs/analysis_plots.png` - Data visualizations

4. **Data Artifacts**:
   - `data/balanced/corpus.jsonl` - Clean, balanced corpus
   - `data/tokenizer/` - Custom WordPiece tokenizer
   - `artifacts/eval/test_embeddings.npz` - Test set embeddings

## Monitoring Progress

The pipeline provides real-time progress bars and logging:

- **Scraping**: Shows pages crawled and queue size
- **Training**: Shows loss, learning rate, and epoch progress
- **Evaluation**: Shows encoding and PCA progress

Training logs are saved in `artifacts/logs/` for later review.

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. Edit `configs/pipeline.yaml`
2. Reduce `batch_size` in train sections
3. Increase `gradient_accumulation_steps`

```yaml
train:
  mlm:
    batch_size: 16  # Reduce from 32
    gradient_accumulation_steps: 4  # Increase from 2
```

### Scraping Issues

If scraping fails:

- Check internet connection
- Some sites may block requests; this is expected
- The pipeline continues with available data

### CUDA Errors

If you don't have a GPU:

1. Training will use CPU (slower but works)
2. Consider reducing model size in `configs/pipeline.yaml`:

```yaml
model:
  hidden_size: 128  # Reduce from 256
  num_hidden_layers: 2  # Reduce from 4
```

## Customization

### Change Topics

Edit `configs/pipeline.yaml` to add/modify topic buckets:

```yaml
topics:
  buckets:
    your_topic:
      keywords: ["keyword1", "keyword2", "keyword3"]
```

### Adjust Scraping Sources

Edit `src/scrape/sources.yaml` to add URLs:

```yaml
seed_urls:
  - url: "https://example.com/mechatronics"
    max_depth: 2
```

### Model Architecture

Modify in `configs/pipeline.yaml`:

```yaml
model:
  hidden_size: 256  # Embedding dimension
  num_hidden_layers: 4  # Transformer layers
  num_attention_heads: 4  # Attention heads
```

## Next Steps

After the pipeline completes:

1. **Explore the 3D visualization**: Open `artifacts/eval/pca_3d.html`
2. **Use the model**: See README.md for inference examples
3. **Fine-tune**: Train on your specific downstream task
4. **Share**: Export the model for others to use

## Getting Help

- Check `README.md` for detailed documentation
- Review `configs/pipeline.yaml` for all configuration options
- Check logs in `artifacts/logs/` for detailed error messages

## Estimated Runtimes

On a typical laptop (CPU only):
- Scraping: 30-60 minutes
- Data processing: 10-20 minutes
- Tokenizer training: 5-10 minutes
- MLM pretraining: 1-2 hours
- TSDAE training: 30-60 minutes
- SimCSE training: 15-30 minutes
- Evaluation: 5-10 minutes

**Total: 4-8 hours**

With GPU (CUDA):
- Training stages: 3-5x faster
- **Total: 2-4 hours**

