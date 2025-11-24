#!/usr/bin/env python3
"""
Train sentence embeddings using TSDAE (Transformer-based Sequential Denoising Auto-Encoder).
TSDAE trains by reconstructing original sentences from noisy versions.
"""

import json
import logging
import random
from pathlib import Path

import click
import torch
import torch.nn as nn
import yaml
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertConfig, get_linear_schedule_with_warmup

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.sentence_pooling import mean_pool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TSDaeDataset(Dataset):
    """Dataset for TSDAE training."""
    
    def __init__(self, file_path: Path, tokenizer: Tokenizer, max_length: int, deletion_prob: float):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.deletion_prob = deletion_prob
        self.sentences = []
        vocab = tokenizer.get_vocab()
        self.cls_id = vocab.get('[CLS]', 1)
        self.sep_id = vocab.get('[SEP]', 2)
        self.pad_id = vocab.get('[PAD]', 0)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    if text and len(text.split()) >= 5:  # Minimum length
                        self.sentences.append(text)
                except Exception:
                    continue
    
    def __len__(self):
        return len(self.sentences)
    
    def _apply_deletion_noise(self, token_ids: list) -> list:
        """Apply deletion noise to tokens."""
        # Keep special tokens
        noisy_ids = []
        for token_id in token_ids:
            # Always keep [CLS], [SEP], [PAD]
            if token_id in [0, 1, 2, 3]:  # Special tokens
                noisy_ids.append(token_id)
            else:
                # Randomly delete with probability
                if random.random() > self.deletion_prob:
                    noisy_ids.append(token_id)
        
        # Ensure at least one token remains (besides special)
        if len(noisy_ids) <= 2:
            return token_ids
        
        return noisy_ids
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        
        # Encode original
        encoding = self.tokenizer.encode(text)
        max_core = max(self.max_length - 2, 0)
        core_ids = encoding.ids[:max_core]
        original_ids = [self.cls_id] + core_ids + [self.sep_id]
        
        # Apply deletion noise
        noisy_ids = self._apply_deletion_noise(original_ids)
        
        return {
            'original_ids': original_ids,
            'noisy_ids': noisy_ids
        }


def collate_fn(batch, pad_token_id=0):
    """Collate with padding."""
    # Get max lengths
    max_original = max(len(item['original_ids']) for item in batch)
    max_noisy = max(len(item['noisy_ids']) for item in batch)
    
    original_ids = []
    original_mask = []
    noisy_ids = []
    noisy_mask = []
    
    for item in batch:
        # Pad original
        orig = item['original_ids']
        orig_pad = orig + [pad_token_id] * (max_original - len(orig))
        orig_mask_item = [1] * len(orig) + [0] * (max_original - len(orig))
        original_ids.append(orig_pad)
        original_mask.append(orig_mask_item)
        
        # Pad noisy
        noisy = item['noisy_ids']
        noisy_pad = noisy + [pad_token_id] * (max_noisy - len(noisy))
        noisy_mask_item = [1] * len(noisy) + [0] * (max_noisy - len(noisy))
        noisy_ids.append(noisy_pad)
        noisy_mask.append(noisy_mask_item)
    
    return {
        'original_ids': torch.tensor(original_ids, dtype=torch.long),
        'original_mask': torch.tensor(original_mask, dtype=torch.long),
        'noisy_ids': torch.tensor(noisy_ids, dtype=torch.long),
        'noisy_mask': torch.tensor(noisy_mask, dtype=torch.long)
    }


class TSDaeModel(nn.Module):
    """TSDAE model: encode noisy input, decode to original."""
    
    def __init__(self, encoder: BertModel, vocab_size: int, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, noisy_ids, noisy_mask, original_ids):
        """
        Encode noisy input, pool to sentence embedding, predict original tokens.
        """
        # Encode noisy input
        encoder_output = self.encoder(input_ids=noisy_ids, attention_mask=noisy_mask)
        
        # Mean pool to get sentence embedding
        sentence_embedding = mean_pool(encoder_output.last_hidden_state, noisy_mask)
        
        # Expand to sequence length for reconstruction
        batch_size = original_ids.size(0)
        seq_len = original_ids.size(1)
        
        # Repeat sentence embedding for each position
        expanded = sentence_embedding.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Decode to vocabulary
        logits = self.decoder(expanded)
        
        return logits


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Train TSDAE sentence embeddings."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    tokenizer_dir = Path(config_data['paths']['tokenizer_dir'])
    splits_dir = Path(config_data['paths']['splits_dir'])
    models_dir = Path(config_data['paths']['models_dir'])
    
    # Load pretrained encoder
    mlm_model_dir = models_dir / 'tiny-bert-mlm' / 'best'
    output_dir = models_dir / 'tsdae-embeddings'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = tokenizer_dir / 'tokenizer.json'
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.get_vocab().get('[PAD]', 0)
    
    # Load encoder
    logger.info(f"Loading pretrained encoder from {mlm_model_dir}")
    encoder = BertModel.from_pretrained(mlm_model_dir)
    hidden_size = encoder.config.hidden_size
    
    # Create TSDAE model
    logger.info("Creating TSDAE model...")
    model = TSDaeModel(encoder, vocab_size, hidden_size)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    train_config = config_data['train']['tsdae']
    max_seq_length = train_config['max_seq_length']
    deletion_prob = train_config['deletion_prob']
    
    # Load datasets
    train_file = splits_dir / 'train.jsonl'
    val_file = splits_dir / 'val.jsonl'
    
    logger.info("Loading datasets...")
    train_dataset = TSDaeDataset(train_file, tokenizer, max_seq_length, deletion_prob)
    val_dataset = TSDaeDataset(val_file, tokenizer, max_seq_length, deletion_prob)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Dataloaders
    batch_size = train_config['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_id),
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_id),
        num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    total_steps = len(train_loader) * train_config['epochs'] // train_config['gradient_accumulation_steps']
    warmup_steps = int(total_steps * train_config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Device - force GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    device = torch.device('cuda')
    logger.info(f"Using device: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Mixed precision
    use_fp16 = train_config.get('fp16', False)
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    logger.info("Starting TSDAE training...")
    
    global_step = 0
    best_val_loss = float('inf')
    grad_accum = train_config['gradient_accumulation_steps']
    
    for epoch in range(train_config['epochs']):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        
        for step, batch in enumerate(pbar):
            noisy_ids = batch['noisy_ids'].to(device)
            noisy_mask = batch['noisy_mask'].to(device)
            original_ids = batch['original_ids'].to(device)
            original_mask = batch['original_mask'].to(device)
            
            # Forward pass
            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits = model(noisy_ids, noisy_mask, original_ids)
                    loss = criterion(logits.view(-1, vocab_size), original_ids.view(-1))
                    loss = loss / grad_accum
                
                scaler.scale(loss).backward()
            else:
                logits = model(noisy_ids, noisy_mask, original_ids)
                loss = criterion(logits.view(-1, vocab_size), original_ids.view(-1))
                loss = loss / grad_accum
                loss.backward()
            
            train_loss += loss.item() * grad_accum
            
            # Update weights
            if (step + 1) % grad_accum == 0:
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                pbar.set_postfix({'loss': train_loss / (step + 1), 'lr': scheduler.get_last_lr()[0]})
                
                # Save checkpoint
                if global_step % train_config['save_steps'] == 0:
                    checkpoint_dir = output_dir / f'checkpoint-{global_step}'
                    checkpoint_dir.mkdir(exist_ok=True)
                    model.encoder.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Validation at end of epoch
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation"):
                logits = model(
                    val_batch['noisy_ids'].to(device),
                    val_batch['noisy_mask'].to(device),
                    val_batch['original_ids'].to(device)
                )
                loss = criterion(logits.view(-1, vocab_size), val_batch['original_ids'].to(device).view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}: Val loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_dir = output_dir / 'best'
            best_model_dir.mkdir(exist_ok=True)
            model.encoder.save_pretrained(best_model_dir)
            logger.info(f"New best model saved! Val loss: {val_loss:.4f}")
    
    # Save final model
    final_dir = output_dir / 'final'
    final_dir.mkdir(exist_ok=True)
    model.encoder.save_pretrained(final_dir)
    logger.info(f"TSDAE training complete! Final model saved to {final_dir}")


if __name__ == '__main__':
    main()

