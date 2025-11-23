#!/usr/bin/env python3
"""
Prepare stratified test set and generate embeddings using the final model.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import BertModel

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.sentence_pooling import mean_pool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_test_samples(test_file: Path, samples_per_bucket: int, seed: int = 42):
    """Prepare stratified test samples."""
    random.seed(seed)
    
    # Collect sentences by topic
    topic_sentences = defaultdict(list)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line)
                text = doc.get('text', '')
                topic = doc.get('topic', 'other')
                
                if text and len(text.split()) >= 5:
                    topic_sentences[topic].append({
                        'text': text,
                        'topic': topic
                    })
            except Exception as e:
                logger.error(f"Error processing line: {e}")
    
    # Sample from each topic
    test_samples = []
    
    for topic, sentences in sorted(topic_sentences.items()):
        if len(sentences) >= samples_per_bucket:
            sampled = random.sample(sentences, samples_per_bucket)
        else:
            sampled = sentences
            logger.warning(f"Topic '{topic}' has only {len(sentences)} samples")
        
        test_samples.extend(sampled)
    
    # Shuffle
    random.shuffle(test_samples)
    
    return test_samples


def encode_sentences(sentences: list, model, tokenizer, device, max_length: int = 256, batch_size: int = 32):
    """Encode sentences to embeddings."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
            batch_texts = [s['text'] for s in sentences[i:i+batch_size]]
            
            # Tokenize
            encodings = [tokenizer.encode(text) for text in batch_texts]
            
            # Prepare batch
            max_len = min(max(len(enc.ids) for enc in encodings), max_length)
            
            input_ids = []
            attention_mask = []
            
            for enc in encodings:
                ids = enc.ids[:max_len]
                mask = [1] * len(ids)
                
                # Pad
                pad_len = max_len - len(ids)
                ids = ids + [0] * pad_len
                mask = mask + [0] * pad_len
                
                input_ids.append(ids)
                attention_mask.append(mask)
            
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
            
            # Encode
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = mean_pool(outputs.last_hidden_state, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Prepare test set and generate embeddings."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    splits_dir = Path(config_data['paths']['splits_dir'])
    test_dir = Path(config_data['paths']['test_dir'])
    tokenizer_dir = Path(config_data['paths']['tokenizer_dir'])
    models_dir = Path(config_data['paths']['models_dir'])
    eval_dir = Path(config_data['paths']['eval_dir'])
    
    test_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load final model
    model_dir = models_dir / 'mecha-embed-v1' / 'best'
    
    logger.info(f"Loading model from {model_dir}")
    model = BertModel.from_pretrained(model_dir)
    
    # Load tokenizer
    tokenizer_path = tokenizer_dir / 'tokenizer.json'
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Prepare test samples
    test_file = splits_dir / 'test.jsonl'
    samples_per_bucket = config_data['eval']['test_samples_per_bucket']
    seed = config_data['eval']['random_seed']
    
    logger.info("Preparing test samples...")
    test_samples = prepare_test_samples(test_file, samples_per_bucket, seed)
    
    logger.info(f"Total test samples: {len(test_samples)}")
    
    # Count topics
    topic_counts = defaultdict(int)
    for sample in test_samples:
        topic_counts[sample['topic']] += 1
    
    print("\n" + "="*60)
    print("TEST SET TOPIC DISTRIBUTION")
    print("="*60)
    for topic in sorted(topic_counts.keys()):
        print(f"{topic:20s}: {topic_counts[topic]:6d}")
    print("="*60)
    
    # Save test samples
    test_samples_file = test_dir / 'test_samples.jsonl'
    with open(test_samples_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Test samples saved to {test_samples_file}")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = encode_sentences(test_samples, model, tokenizer, device)
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    # Save embeddings and metadata
    output_file = eval_dir / 'test_embeddings.npz'
    
    texts = [s['text'] for s in test_samples]
    topics = [s['topic'] for s in test_samples]
    
    np.savez(
        output_file,
        embeddings=embeddings,
        texts=texts,
        topics=topics
    )
    
    logger.info(f"Embeddings saved to {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EMBEDDING STATISTICS")
    print("="*60)
    print(f"Shape: {embeddings.shape}")
    print(f"Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"Std norm: {np.linalg.norm(embeddings, axis=1).std():.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

