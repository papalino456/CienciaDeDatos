#!/usr/bin/env python3
"""
Clean and deduplicate the extracted dataset.
"""

import json
import logging
import re
from pathlib import Path

import click
import yaml
from datasketch import MinHash, MinHashLSH
from langdetect import detect, LangDetectException
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_english(text: str, confidence_threshold: float = 0.9) -> bool:
    """Check if text is English with confidence."""
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False


def normalize_text(text: str) -> str:
    """Normalize whitespace and clean text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove code blocks (common pattern)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,;:!?()\-\'\"]+', ' ', text)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    return text.strip()


def split_into_sentences(text: str) -> list:
    """Simple sentence splitter."""
    # Split on period, exclamation, question mark followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def get_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Create MinHash for text."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for word in words:
        m.update(word.encode('utf8'))
    return m


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Clean and deduplicate the dataset."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    interim_dir = Path(config_data['paths']['interim_data'])
    clean_dir = Path(config_data['paths']['clean_data'])
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = interim_dir / 'extracted.jsonl'
    output_file = clean_dir / 'cleaned.jsonl'
    
    # Config parameters
    min_sent_len = config_data['clean']['min_sentence_length']
    max_sent_len = config_data['clean']['max_sentence_length']
    min_words = config_data['clean']['min_word_count']
    max_words = config_data['clean']['max_word_count']
    dedup_threshold = config_data['clean']['dedup_threshold']
    english_confidence = config_data['clean']['english_confidence']
    
    logger.info("Cleaning dataset...")
    
    # Track URLs and near-duplicates
    seen_urls = set()
    lsh = MinHashLSH(threshold=dedup_threshold, num_perm=128)
    
    stats = {
        'total': 0,
        'url_duplicates': 0,
        'non_english': 0,
        'too_short': 0,
        'too_long': 0,
        'near_duplicates': 0,
        'kept': 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        for line in tqdm(in_f, desc="Cleaning documents"):
            stats['total'] += 1
            
            try:
                doc = json.loads(line)
                url = doc.get('url', '')
                text = doc.get('text', '')
                
                # Check URL duplicate
                if url in seen_urls:
                    stats['url_duplicates'] += 1
                    continue
                seen_urls.add(url)
                
                # Check language
                if not is_english(text, english_confidence):
                    stats['non_english'] += 1
                    continue
                
                # Normalize text
                text = normalize_text(text)
                
                # Split into sentences
                sentences = split_into_sentences(text)
                
                # Filter sentences
                valid_sentences = []
                for sent in sentences:
                    word_count = len(sent.split())
                    char_count = len(sent)
                    
                    if char_count < min_sent_len:
                        continue
                    if char_count > max_sent_len:
                        continue
                    if word_count < min_words:
                        continue
                    if word_count > max_words:
                        continue
                    
                    valid_sentences.append(sent)
                
                if not valid_sentences:
                    stats['too_short'] += 1
                    continue
                
                # Rejoin filtered text
                filtered_text = ' '.join(valid_sentences)
                
                # Check for near-duplicates using MinHash
                minhash = get_minhash(filtered_text)
                
                # Query LSH for similar documents
                result = lsh.query(minhash)
                if result:
                    stats['near_duplicates'] += 1
                    continue
                
                # Insert into LSH
                lsh.insert(url, minhash)
                
                # Save cleaned document
                cleaned_doc = {
                    'url': url,
                    'title': doc.get('title', ''),
                    'text': filtered_text,
                    'sentences': valid_sentences,
                    'sentence_count': len(valid_sentences)
                }
                
                out_f.write(json.dumps(cleaned_doc, ensure_ascii=False) + '\n')
                stats['kept'] += 1
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATA CLEANING SUMMARY")
    print("="*60)
    print(f"Total documents: {stats['total']}")
    print(f"URL duplicates: {stats['url_duplicates']}")
    print(f"Non-English: {stats['non_english']}")
    print(f"Too short/long: {stats['too_short']}")
    print(f"Near-duplicates: {stats['near_duplicates']}")
    print(f"Kept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
    print("="*60)
    
    logger.info(f"Cleaned data saved to {output_file}")
    
    # Save stats
    logs_dir = Path(config_data['paths']['logs_dir'])
    with open(logs_dir / 'cleaning_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()

