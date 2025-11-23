#!/usr/bin/env python3
"""
Train sentence embeddings using SimCSE (Simple Contrastive Learning of Sentence Embeddings).
Unsupervised version: uses dropout as noise to create positive pairs.
"""

import json
import logging
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.sentence_pooling import mean_pool, cls_pool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimCSEDataset(Dataset):
    """Dataset for SimCSE training."""
    
    def __init__(self, file_path: Path, tokenizer: Tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    if text and len(text.split()) >= 5:
                        self.sentences.append(text)
                except Exception:
                    continue
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        encoding = self.tokenizer.encode(text)
        
        input_ids = encoding.ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def collate_fn(batch, pad_token_id=0):
    """Collate with padding."""
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_mask = []
    
    for item in batch:
        ids = item['input_ids']
        mask = item['attention_mask']
        
        # Pad
        pad_len = max_len - len(ids)
        ids = ids + [pad_token_id] * pad_len
        mask = mask + [0] * pad_len
        
        input_ids.append(ids)
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }


def simcse_loss(embeddings1, embeddings2, temperature=0.05):
    """
    Compute SimCSE contrastive loss.
    
    Args:
        embeddings1: [batch_size, hidden_size] - first view
        embeddings2: [batch_size, hidden_size] - second view (with different dropout)
        temperature: temperature for scaling
    
    Returns:
        loss: scalar
    """
    batch_size = embeddings1.size(0)
    
    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    
    # Compute similarity matrix: [batch_size, batch_size]
    # Each row i: similarity of embeddings1[i] with all embeddings2[j]
    sim_matrix = torch.matmul(embeddings1, embeddings2.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=embeddings1.device)
    
    # Cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


class SimCSEModel(nn.Module):
    """SimCSE model wrapper."""
    
    def __init__(self, encoder: BertModel, pooling_mode='mean'):
        super().__init__()
        self.encoder = encoder
        self.pooling_mode = pooling_mode
    
    def forward(self, input_ids, attention_mask):
        """Forward pass with pooling."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.pooling_mode == 'mean':
            return mean_pool(outputs.last_hidden_state, attention_mask)
        elif self.pooling_mode == 'cls':
            return cls_pool(outputs.last_hidden_state)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_mode}")


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Train SimCSE sentence embeddings."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    tokenizer_dir = Path(config_data['paths']['tokenizer_dir'])
    splits_dir = Path(config_data['paths']['splits_dir'])
    models_dir = Path(config_data['paths']['models_dir'])
    
    # Load TSDAE model as starting point
    tsdae_model_dir = models_dir / 'tsdae-embeddings' / 'best'
    output_dir = models_dir / 'mecha-embed-v1'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = tokenizer_dir / 'tokenizer.json'
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Load encoder
    logger.info(f"Loading TSDAE encoder from {tsdae_model_dir}")
    encoder = BertModel.from_pretrained(tsdae_model_dir)
    
    # Training config
    train_config = config_data['train']['simcse']
    pooling_mode = 'mean'  # Use mean pooling
    
    # Create SimCSE model
    logger.info("Creating SimCSE model...")
    model = SimCSEModel(encoder, pooling_mode)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    train_file = splits_dir / 'train.jsonl'
    val_file = splits_dir / 'val.jsonl'
    
    max_seq_length = train_config['max_seq_length']
    
    logger.info("Loading datasets...")
    train_dataset = SimCSEDataset(train_file, tokenizer, max_seq_length)
    val_dataset = SimCSEDataset(val_file, tokenizer, max_seq_length)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Dataloaders
    batch_size = train_config['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
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
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Temperature
    temperature = train_config['temperature']
    
    # Mixed precision
    use_fp16 = train_config.get('fp16', False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    logger.info(f"Starting SimCSE training with temperature={temperature}...")
    
    global_step = 0
    best_val_loss = float('inf')
    grad_accum = train_config['gradient_accumulation_steps']
    
    for epoch in range(train_config['epochs']):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass twice with different dropout (creates positive pairs)
            if use_fp16:
                with torch.cuda.amp.autocast():
                    embeddings1 = model(input_ids, attention_mask)
                    embeddings2 = model(input_ids, attention_mask)
                    
                    loss = simcse_loss(embeddings1, embeddings2, temperature)
                    loss = loss / grad_accum
                
                scaler.scale(loss).backward()
            else:
                embeddings1 = model(input_ids, attention_mask)
                embeddings2 = model(input_ids, attention_mask)
                
                loss = simcse_loss(embeddings1, embeddings2, temperature)
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
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                
                # Two forward passes
                embeddings1 = model(input_ids, attention_mask)
                embeddings2 = model(input_ids, attention_mask)
                
                loss = simcse_loss(embeddings1, embeddings2, temperature)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}: Val loss = {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_dir = output_dir / 'best'
            best_model_dir.mkdir(exist_ok=True)
            model.encoder.save_pretrained(best_model_dir)
            
            # Save pooling config
            pooling_config = {'pooling_mode': pooling_mode}
            with open(best_model_dir / 'pooling_config.json', 'w') as f:
                json.dump(pooling_config, f, indent=2)
            
            logger.info(f"New best model saved! Val loss: {val_loss:.4f}")
    
    # Save final model
    final_dir = output_dir / 'final'
    final_dir.mkdir(exist_ok=True)
    model.encoder.save_pretrained(final_dir)
    
    with open(final_dir / 'pooling_config.json', 'w') as f:
        json.dump({'pooling_mode': pooling_mode}, f, indent=2)
    
    logger.info(f"SimCSE training complete! Final model saved to {final_dir}")
    logger.info(f"Best model saved to {output_dir / 'best'}")


if __name__ == '__main__':
    main()

