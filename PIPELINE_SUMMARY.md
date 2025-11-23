# Pipeline Implementation Summary

## Overview

A complete end-to-end ML pipeline for training domain-specific sentence embeddings from scratch for the mechatronics domain, implementing state-of-the-art low-data techniques.

## Implementation Details

### 1. Web Scraping & Data Collection

**Files**: `src/scrape/crawl.py`, `src/scrape/extract.py`, `src/scrape/sources.yaml`

**Features**:
- **Polite crawling**: Respects robots.txt, implements rate limiting (1s between requests)
- **Async architecture**: Uses httpx for concurrent requests
- **Multiple sources**:
  - Wikipedia categories (Mechatronics, Control Engineering, Robotics, etc.)
  - arXiv API (cs.RO, eess.SY, cs.AI categories)
  - Technical documentation (ROS wiki, etc.)
- **Error handling**: Retries with exponential backoff, timeout handling
- **Deduplication**: URL-based deduplication during crawl
- **Text extraction**: Uses trafilatura for main content extraction

**Output**: Raw HTML and extracted text in JSONL format

### 2. Data Analysis & Cleaning

**Files**: `src/data/analyze.py`, `src/data/clean.py`

**Analysis**:
- Language detection (English filtering at 90% confidence)
- Document length statistics (chars, words, sentences)
- Topic categorization using keyword matching
- Duplicate detection (exact and near-duplicate via MinHash LSH)
- Visualization (length distributions, topic histograms)

**Cleaning**:
- Text normalization (whitespace, URL removal, code block removal)
- Sentence splitting with regex
- Length filtering (10-512 chars, 3-100 words per sentence)
- Near-duplicate removal (85% Jaccard similarity threshold)
- English-only filtering
- Saves cleaning statistics

**Output**: Cleaned sentences in JSONL with metadata

### 3. Topic Balancing

**File**: `src/data/balance.py`

**Topic Buckets** (8 domains):
1. Control (PID, feedback, servo)
2. Robotics (manipulators, path planning)
3. Sensors (encoders, LiDAR, IMU)
4. Actuators (motors, hydraulics)
5. PLC (programmable logic, SCADA)
6. Embedded (microcontrollers, firmware, RTOS)
7. Kinematics (forward/inverse, Jacobians)
8. Dynamics (forces, torques, equations of motion)

**Balancing**:
- Keyword-based categorization
- Stratified sampling (target: 2000 samples/bucket, min: 500)
- 80/10/10 train/val/test splits
- Preserves topic distribution across splits

**Output**: Balanced corpus and splits in JSONL

### 4. Tokenizer Training

**File**: `src/tokenizer/train_tokenizer.py`

**Configuration**:
- Algorithm: WordPiece (BERT-style)
- Vocabulary size: 16,000 tokens
- Normalization: Lowercase, NFD, strip accents
- Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
- Min frequency: 2

**Process**:
- Trains from scratch on balanced corpus
- Domain-specific vocabulary (mechatronics terms)
- Saves tokenizer.json and vocab.txt

**Output**: Custom WordPiece tokenizer for mechatronics

### 5. MLM Pretraining

**File**: `src/train/pretrain_mlm.py`

**Architecture** (Tiny BERT):
- Hidden size: 256
- Layers: 4
- Attention heads: 4
- Intermediate size: 1024
- Max position: 512
- Total parameters: ~3.5M

**Training**:
- Objective: Masked Language Modeling (15% masking)
- Optimizer: AdamW (lr=5e-4, weight_decay=0.01)
- Scheduler: Cosine with warmup (500 steps)
- Batch size: 32 with gradient accumulation (2 steps)
- Max steps: 10,000 (or 5 epochs)
- Mixed precision: FP16 (if CUDA available)
- Checkpointing: Every 1000 steps
- Validation: Every 500 steps

**Purpose**: Initialize encoder with domain knowledge before sentence embedding training

**Output**: Pretrained encoder weights

### 6. TSDAE Training

**File**: `src/train/train_tsdae.py`

**Method**: Transformer-based Sequential Denoising Auto-Encoder

**Approach**:
- **Noise**: Token deletion (60% probability per token)
- **Objective**: Reconstruct original sentence from noisy input
- **Architecture**: Encoder → Mean pool → Decoder (linear to vocab)
- **Loss**: Cross-entropy on token predictions

**Training**:
- Starts from MLM pretrained weights
- Batch size: 16 with gradient accumulation (4 steps)
- Learning rate: 3e-5
- Epochs: 3
- Max sequence length: 256
- Warmup ratio: 0.1

**Why TSDAE**:
- Effective for unsupervised learning with small data
- Forces encoder to capture sentence-level semantics
- Noise regularization improves robustness

**Output**: Encoder with sentence-level representations

### 7. SimCSE Training

**File**: `src/train/train_simcse.py`

**Method**: Simple Contrastive Learning of Sentence Embeddings

**Approach**:
- **Unsupervised**: Uses dropout as noise source
- **Positive pairs**: Same sentence, different dropout masks
- **Negative pairs**: Other sentences in batch (in-batch negatives)
- **Loss**: InfoNCE (cross-entropy on similarity matrix)
- **Temperature**: 0.05

**Training**:
- Starts from TSDAE weights
- Batch size: 64 (larger for more negatives)
- Learning rate: 3e-5
- Epochs: 1 (quick refinement)
- Pooling: Mean pooling with L2 normalization

**Why SimCSE**:
- State-of-the-art unsupervised sentence embeddings
- Extremely simple: just two forward passes with dropout
- In-batch negatives: O(n²) pairs from n samples
- Sharpens semantic neighborhoods

**Output**: Final sentence embedding model (mecha-embed-v1)

### 8. Evaluation & Visualization

**Files**: `src/eval/prepare_testset.py`, `src/eval/visualize_pca.py`

**Test Set Preparation**:
- Stratified sampling: 125 samples per topic bucket
- Total: ~1000 test sentences
- Balanced across all 8 topics
- Generate embeddings for all test samples

**PCA Visualization**:
- Reduce from 256D to 3D using PCA
- Calculate explained variance per component
- Create interactive 3D scatter plot (Plotly)
- Color by topic
- Hover shows sentence text
- Export to HTML for browser viewing

**Metrics**:
- Explained variance ratio
- Embedding norms (should be ~1 after L2 norm)
- Topic clustering (visual inspection in 3D plot)

**Output**: 
- `test_embeddings.npz` (embeddings, texts, topics)
- `pca_3d.html` (interactive visualization)
- `pca_stats.json` (variance explained)

## Key Design Decisions

### 1. From-Scratch Training
- **Why**: Truly domain-specific, not transfer learning
- **Challenge**: Requires careful regularization and low-data techniques
- **Solution**: TSDAE + SimCSE pipeline

### 2. Tiny Architecture (256 hidden, 4 layers)
- **Why**: Faster training, less overfitting on small data
- **Trade-off**: Lower capacity, but sufficient for domain task
- **Benefit**: ~3.5M params vs ~110M for BERT-base

### 3. Three-Stage Training
- **MLM**: Domain knowledge (vocabulary, syntax)
- **TSDAE**: Sentence-level semantics (denoising)
- **SimCSE**: Semantic neighborhoods (contrastive)
- **Why**: Progressive refinement from word → sentence → similarity

### 4. Unsupervised Only
- **Why**: No labeled data available
- **Techniques**: Dropout noise (SimCSE), deletion noise (TSDAE)
- **Benefit**: Scalable to any domain with raw text

### 5. Topic Balancing
- **Why**: Prevent model bias toward common topics
- **Method**: Stratified sampling across 8 buckets
- **Benefit**: Fair representation of all mechatronics subfields

## Low-Data Techniques Summary

1. **TSDAE (Denoising)**: Learns sentence representations by reconstructing from noisy input
2. **SimCSE (Contrastive)**: Learns by distinguishing similar vs dissimilar sentences
3. **In-batch negatives**: Efficient contrastive learning (O(n²) pairs from n samples)
4. **Dropout as augmentation**: Creates positive pairs without data augmentation
5. **Progressive training**: MLM → TSDAE → SimCSE builds representations incrementally
6. **Small model**: Reduces overfitting risk with limited data
7. **Mixed precision**: FP16 enables larger batches with same memory
8. **Gradient accumulation**: Simulates larger batches for stable training

## Pipeline Flexibility

**Configurable via `configs/pipeline.yaml`**:
- Scraping limits and sources
- Cleaning thresholds
- Topic keywords and balancing targets
- Model architecture (hidden size, layers, heads)
- Training hyperparameters (batch, lr, epochs)
- Evaluation settings

**Skip flags in PowerShell script**:
- `-SkipScrape`: Use existing raw data
- `-SkipClean`: Use existing cleaned data
- `-SkipTokenizer`: Use existing tokenizer
- `-SkipMLM`: Use existing pretrained encoder
- `-SkipTSDAE`: Use existing TSDAE model
- `-SkipSimCSE`: Use existing final model
- `-SkipEval`: Skip evaluation

## Expected Results

**Dataset**:
- ~5000-10000 scraped documents
- ~20000-40000 sentences after cleaning
- ~16000 balanced sentences (2000 per topic)

**Model**:
- 256-dimensional sentence embeddings
- L2 normalized (unit length)
- Captures semantic similarity for mechatronics text

**Visualization**:
- Clear topic clustering in 3D PCA
- ~40-60% variance explained by 3 components
- Interactive exploration in browser

## Technical Stack

- **Web**: httpx, BeautifulSoup, trafilatura
- **NLP**: tokenizers (HuggingFace), langdetect, NLTK
- **ML/DL**: PyTorch, transformers, scikit-learn
- **Deduplication**: datasketch (MinHash LSH)
- **Visualization**: Plotly, Matplotlib
- **Config**: PyYAML
- **Platform**: Windows PowerShell scripts

## References & Inspiration

1. **TSDAE**: Wang et al., EMNLP 2021 - "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning"
2. **SimCSE**: Gao et al., EMNLP 2021 - "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
3. **Sentence-BERT**: Reimers & Gurevych, EMNLP 2019 - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
4. **In-batch negatives**: Henderson et al., EMNLP 2017 - "Efficient Natural Language Response Suggestion for Smart Reply"

## Limitations & Future Work

**Current Limitations**:
- English only (could extend to multilingual)
- Single domain (mechatronics)
- No supervised fine-tuning (could add task-specific layer)
- Static embeddings (no contextualization beyond sentence)

**Future Enhancements**:
- Add supervised training for specific tasks (classification, retrieval)
- Implement hard negative mining for better contrastive learning
- Add data augmentation (back-translation, paraphrasing)
- Scale to larger models once more data available
- Add cross-encoder re-ranking for retrieval tasks
- Implement knowledge distillation from larger models

## Success Metrics

The pipeline successfully:
1. ✅ Scrapes and builds a mechatronics dataset from web sources
2. ✅ Analyzes and cleans data (language detection, deduplication, balancing)
3. ✅ Trains a custom tokenizer from scratch
4. ✅ Pretrains a tiny BERT with MLM from scratch
5. ✅ Trains sentence embeddings using TSDAE + SimCSE (low-data techniques)
6. ✅ Generates test embeddings and 3D PCA visualization
7. ✅ Provides complete, reproducible pipeline with documentation
8. ✅ Implements all components without using pre-trained embeddings

