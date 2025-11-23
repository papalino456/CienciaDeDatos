#!/usr/bin/env python3
"""
Balance dataset across topic buckets and create train/val/test splits.
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import click
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def categorize_sentences(sentences: list, text: str, topic_config: dict) -> dict:
    """Categorize sentences by topic."""
    text_lower = text.lower()
    
    # Count topic matches
    topic_matches = defaultdict(int)
    for topic, info in topic_config.items():
        keywords = info['keywords']
        for keyword in keywords:
            if keyword in text_lower:
                topic_matches[topic] += 1
    
    # Assign primary topic (most matches)
    if topic_matches:
        primary_topic = max(topic_matches.items(), key=lambda x: x[1])[0]
    else:
        primary_topic = 'other'
    
    return {
        'primary_topic': primary_topic,
        'topic_matches': dict(topic_matches)
    }


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Balance dataset across topics and create splits."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    clean_dir = Path(config_data['paths']['clean_data'])
    balanced_dir = Path(config_data['paths']['balanced_data'])
    splits_dir = Path(config_data['paths']['splits_dir'])
    balanced_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = clean_dir / 'cleaned.jsonl'
    
    topic_config = config_data['topics']['buckets']
    target_per_bucket = config_data['topics']['target_samples_per_bucket']
    min_per_bucket = config_data['topics']['min_samples_per_bucket']
    
    random.seed(config_data['eval']['random_seed'])
    
    logger.info("Categorizing sentences by topic...")
    
    # Collect sentences by topic
    topic_sentences = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Categorizing"):
            try:
                doc = json.loads(line)
                text = doc.get('text', '')
                sentences = doc.get('sentences', [])
                
                if not sentences:
                    continue
                
                # Categorize document
                categorization = categorize_sentences(sentences, text, topic_config)
                primary_topic = categorization['primary_topic']
                
                # Add sentences to topic bucket
                for sent in sentences:
                    topic_sentences[primary_topic].append({
                        'text': sent,
                        'topic': primary_topic,
                        'url': doc.get('url', '')
                    })
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
    
    # Print topic statistics
    print("\n" + "="*60)
    print("TOPIC DISTRIBUTION (before balancing)")
    print("="*60)
    for topic in sorted(topic_sentences.keys()):
        count = len(topic_sentences[topic])
        print(f"{topic:20s}: {count:6d} sentences")
    print("="*60)
    
    # Balance topics
    balanced_sentences = []
    
    for topic in sorted(topic_sentences.keys()):
        sentences = topic_sentences[topic]
        count = len(sentences)
        
        if count < min_per_bucket:
            logger.warning(f"Topic '{topic}' has only {count} sentences (min: {min_per_bucket})")
            # Use all available
            balanced_sentences.extend(sentences)
        elif count > target_per_bucket:
            # Randomly sample
            sampled = random.sample(sentences, target_per_bucket)
            balanced_sentences.extend(sampled)
        else:
            # Use all
            balanced_sentences.extend(sentences)
    
    # Shuffle
    random.shuffle(balanced_sentences)
    
    logger.info(f"Total balanced sentences: {len(balanced_sentences)}")
    
    # Save balanced corpus
    corpus_file = balanced_dir / 'corpus.jsonl'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for item in balanced_sentences:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Balanced corpus saved to {corpus_file}")
    
    # Create train/val/test splits (80/10/10)
    total = len(balanced_sentences)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    
    train_data = balanced_sentences[:train_size]
    val_data = balanced_sentences[train_size:train_size+val_size]
    test_data = balanced_sentences[train_size+val_size:]
    
    # Save splits
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        split_file = splits_dir / f'{split_name}.jsonl'
        with open(split_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"{split_name.capitalize()} split: {len(split_data)} sentences -> {split_file}")
    
    # Save topic distribution for splits
    for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        topic_dist = defaultdict(int)
        for item in split_data:
            topic_dist[item['topic']] += 1
        
        print(f"\n{split_name.upper()} split topic distribution:")
        for topic in sorted(topic_dist.keys()):
            print(f"  {topic:20s}: {topic_dist[topic]:6d}")
    
    print("\n" + "="*60)
    print("BALANCING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

