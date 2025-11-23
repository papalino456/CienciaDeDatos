#!/usr/bin/env python3
"""
Train a WordPiece tokenizer from scratch on the mechatronics corpus.
"""

import json
import logging
from pathlib import Path

import click
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_training_corpus(file_path: Path, batch_size: int = 1000):
    """Generator for training corpus."""
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line)
                text = doc.get('text', '')
                if text:
                    batch.append(text)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            except Exception as e:
                logger.error(f"Error reading line: {e}")
    
    if batch:
        yield batch


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Train WordPiece tokenizer."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    balanced_dir = Path(config_data['paths']['balanced_data'])
    tokenizer_dir = Path(config_data['paths']['tokenizer_dir'])
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_file = balanced_dir / 'corpus.jsonl'
    
    if not corpus_file.exists():
        logger.error(f"Corpus file not found: {corpus_file}")
        return
    
    # Tokenizer config
    vocab_size = config_data['tokenizer']['vocab_size']
    min_frequency = config_data['tokenizer']['min_frequency']
    special_tokens = config_data['tokenizer']['special_tokens']
    
    logger.info("Training WordPiece tokenizer...")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Min frequency: {min_frequency}")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    
    # Add normalizers
    if config_data['tokenizer']['lowercase']:
        normalizers = [NFD(), Lowercase()]
    else:
        normalizers = [NFD()]
    
    if config_data['tokenizer']['strip_accents']:
        normalizers.append(StripAccents())
    
    tokenizer.normalizer = Sequence(normalizers)
    
    # Add pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    
    # Trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train on corpus
    logger.info("Reading corpus and training...")
    
    # First, count lines for progress
    with open(corpus_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    logger.info(f"Training on {total_lines} sentences")
    
    # Train
    training_corpus = get_training_corpus(corpus_file)
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)
    
    logger.info("Training complete!")
    
    # Save tokenizer
    output_path = tokenizer_dir / 'tokenizer.json'
    tokenizer.save(str(output_path))
    logger.info(f"Tokenizer saved to {output_path}")
    
    # Save config
    tokenizer_config = {
        'vocab_size': vocab_size,
        'model_type': 'wordpiece',
        'special_tokens': special_tokens,
        'lowercase': config_data['tokenizer']['lowercase'],
        'strip_accents': config_data['tokenizer']['strip_accents']
    }
    
    config_path = tokenizer_dir / 'tokenizer_config.json'
    with open(config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    # Test tokenizer
    test_sentences = [
        "The PID controller regulates the servo motor position.",
        "Robot kinematics involves forward and inverse transformations.",
        "Sensors measure physical quantities and convert them to electrical signals."
    ]
    
    print("\n" + "="*60)
    print("TOKENIZER TEST")
    print("="*60)
    
    for sent in test_sentences:
        encoded = tokenizer.encode(sent)
        print(f"\nOriginal: {sent}")
        print(f"Tokens: {encoded.tokens}")
        print(f"IDs: {encoded.ids}")
    
    print("="*60)
    
    # Get vocabulary statistics
    vocab = tokenizer.get_vocab()
    logger.info(f"Final vocabulary size: {len(vocab)}")
    
    # Save vocab file
    vocab_file = tokenizer_dir / 'vocab.txt'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{token}\n")
    
    logger.info(f"Vocabulary saved to {vocab_file}")


if __name__ == '__main__':
    main()

