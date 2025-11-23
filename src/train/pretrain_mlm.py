#!/usr/bin/env python3
"""
Pretrain a tiny BERT model from scratch using Masked Language Modeling (MLM).
"""

import json
import logging
from pathlib import Path

import click
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    AdamW,
    get_cosine_schedule_with_warmup,
    DataCollatorForLanguageModeling
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentenceDataset(Dataset):
    """Dataset for sentences."""
    
    def __init__(self, file_path: Path, tokenizer: Tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    if text:
                        self.sentences.append(text)
                except Exception:
                    continue
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        encoding = self.tokenizer.encode(text)
        
        # Truncate to max length
        input_ids = encoding.ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function with padding."""
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    attention_mask = []
    
    for item in batch:
        ids = item['input_ids']
        mask = item['attention_mask']
        
        # Pad
        padding_length = max_len - len(ids)
        ids = torch.cat([ids, torch.zeros(padding_length, dtype=torch.long)])
        mask = torch.cat([mask, torch.zeros(padding_length, dtype=torch.long)])
        
        input_ids.append(ids)
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask)
    }


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Pretrain BERT with MLM."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    tokenizer_dir = Path(config_data['paths']['tokenizer_dir'])
    splits_dir = Path(config_data['paths']['splits_dir'])
    models_dir = Path(config_data['paths']['models_dir'])
    
    output_dir = models_dir / 'tiny-bert-mlm'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = tokenizer_dir / 'tokenizer.json'
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Create BERT config
    model_config = config_data['model']
    bert_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=model_config['hidden_size'],
        num_hidden_layers=model_config['num_hidden_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        max_position_embeddings=model_config['max_position_embeddings'],
        hidden_dropout_prob=model_config['hidden_dropout_prob'],
        attention_probs_dropout_prob=model_config['attention_probs_dropout_prob']
    )
    
    # Initialize model from scratch
    logger.info("Initializing BERT model from scratch...")
    model = BertForMaskedLM(bert_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    train_file = splits_dir / 'train.jsonl'
    val_file = splits_dir / 'val.jsonl'
    
    logger.info("Loading datasets...")
    train_dataset = SentenceDataset(train_file, tokenizer, model_config['max_position_embeddings'])
    val_dataset = SentenceDataset(val_file, tokenizer, model_config['max_position_embeddings'])
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Training config
    train_config = config_data['train']['mlm']
    batch_size = train_config['batch_size']
    grad_accum = train_config['gradient_accumulation_steps']
    
    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=train_config['mlm_probability']
    )
    
    # Dataloaders
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
    
    total_steps = min(
        train_config['max_steps'],
        len(train_loader) * train_config['epochs'] // grad_accum
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Mixed precision
    use_fp16 = train_config.get('fp16', False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    logger.info("Starting MLM pretraining...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(train_config['epochs']):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
        
        for step, batch in enumerate(pbar):
            # Apply MLM masking
            masked_batch = data_collator([
                {'input_ids': batch['input_ids'][i].tolist()}
                for i in range(batch['input_ids'].size(0))
            ])
            
            input_ids = masked_batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = masked_batch['labels'].to(device)
            
            # Forward pass
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / grad_accum
                
                scaler.scale(loss).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / grad_accum
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
                    model.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
                
                # Validation
                if global_step % train_config['eval_steps'] == 0:
                    model.eval()
                    val_loss = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            masked_val = data_collator([
                                {'input_ids': val_batch['input_ids'][i].tolist()}
                                for i in range(val_batch['input_ids'].size(0))
                            ])
                            
                            val_outputs = model(
                                input_ids=masked_val['input_ids'].to(device),
                                attention_mask=val_batch['attention_mask'].to(device),
                                labels=masked_val['labels'].to(device)
                            )
                            val_loss += val_outputs.loss.item()
                    
                    val_loss /= len(val_loader)
                    logger.info(f"Step {global_step}: Val loss = {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_dir = output_dir / 'best'
                        best_model_dir.mkdir(exist_ok=True)
                        model.save_pretrained(best_model_dir)
                        logger.info(f"New best model saved! Val loss: {val_loss:.4f}")
                    
                    model.train()
                
                if global_step >= total_steps:
                    break
        
        if global_step >= total_steps:
            break
    
    # Save final model
    final_dir = output_dir / 'final'
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    logger.info(f"Training complete! Final model saved to {final_dir}")


if __name__ == '__main__':
    main()

