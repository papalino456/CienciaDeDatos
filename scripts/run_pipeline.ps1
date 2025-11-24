# PowerShell script to run the complete ML pipeline
# Run from project root: .\scripts\run_pipeline.ps1

param(
    [switch]$SkipScrape,
    [switch]$SkipClean,
    [switch]$SkipTokenizer,
    [switch]$SkipMLM,
    [switch]$SkipTSDAE,
    [switch]$SkipSimCSE,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Mechatronics Embeddings Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$config = "configs/pipeline.yaml"

# Step 1: Scraping
if (-not $SkipScrape) {
    Write-Host "`n[1/10] Web Scraping..." -ForegroundColor Yellow
    python src/scrape/crawl.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Crawling failed" }
    
    Write-Host "`n[2/10] Text Extraction..." -ForegroundColor Yellow
    python src/scrape/extract.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Extraction failed" }
} else {
    Write-Host "`n[1-2/10] Skipping scraping steps" -ForegroundColor Gray
}

# Step 2: Data Analysis and Cleaning
if (-not $SkipClean) {
    Write-Host "`n[3/10] Analyzing Dataset..." -ForegroundColor Yellow
    python src/data/analyze.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed" }
    
    Write-Host "`n[4/10] Cleaning Dataset..." -ForegroundColor Yellow
    python src/data/clean.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Cleaning failed" }
    
    Write-Host "`n[5/10] Balancing Topics..." -ForegroundColor Yellow
    python src/data/balance.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Balancing failed" }
} else {
    Write-Host "`n[3-5/10] Skipping data cleaning steps" -ForegroundColor Gray
}

# Step 3: Tokenizer Training
if (-not $SkipTokenizer) {
    Write-Host "`n[6/10] Training Tokenizer..." -ForegroundColor Yellow
    python src/tokenizer/train_tokenizer.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Tokenizer training failed" }
} else {
    Write-Host "`n[6/10] Skipping tokenizer training" -ForegroundColor Gray
}

# Step 4: MLM Pretraining
if (-not $SkipMLM) {
    Write-Host "`n[7/10] Pretraining BERT with MLM..." -ForegroundColor Yellow
    python src/train/pretrain_mlm.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "MLM pretraining failed" }
} else {
    Write-Host "`n[7/10] Skipping MLM pretraining" -ForegroundColor Gray
}

# Step 5: TSDAE Training
if (-not $SkipTSDAE) {
    Write-Host "`n[8/10] Training TSDAE..." -ForegroundColor Yellow
    python src/train/train_tsdae.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "TSDAE training failed" }
} else {
    Write-Host "`n[8/10] Skipping TSDAE training" -ForegroundColor Gray
}

# Step 6: SimCSE Training
if (-not $SkipSimCSE) {
    Write-Host "`n[9/10] Training SimCSE..." -ForegroundColor Yellow
    python src/train/train_simcse.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "SimCSE training failed" }
} else {
    Write-Host "`n[9/10] Skipping SimCSE training" -ForegroundColor Gray
}

# Step 7: Evaluation and Visualization
if (-not $SkipEval) {
    Write-Host "`n[10/10] Preparing Test Set and Generating Embeddings..." -ForegroundColor Yellow
    python src/eval/prepare_testset.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Test set preparation failed" }
    
    Write-Host "`nVisualizing with 3D PCA..." -ForegroundColor Yellow
    python src/eval/visualize_pca.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "PCA visualization failed" }
    
    Write-Host "`nVisualizing with 3D UMAP..." -ForegroundColor Yellow
    python src/eval/visualize_umap.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "UMAP visualization failed" }
} else {
    Write-Host "`n[10/10] Skipping evaluation" -ForegroundColor Gray
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nOutputs:" -ForegroundColor Cyan
Write-Host "  - Final model: artifacts/models/mecha-embed-v1/best/" -ForegroundColor White
Write-Host "  - 3D PCA visualization: artifacts/eval/pca_3d.html" -ForegroundColor White
Write-Host "  - 3D UMAP visualization: artifacts/eval/umap_3d.html" -ForegroundColor White
Write-Host "  - Logs: artifacts/logs/" -ForegroundColor White
Write-Host "`nOpen artifacts/eval/pca_3d.html or artifacts/eval/umap_3d.html in your browser to view the embeddings!" -ForegroundColor Yellow

