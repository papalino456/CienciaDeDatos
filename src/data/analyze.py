#!/usr/bin/env python3
"""
Analyze the extracted dataset for quality, language, and topic distribution.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from langdetect import detect, LangDetectException
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language with error handling."""
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def analyze_text_stats(text: str) -> dict:
    """Get basic text statistics."""
    sentences = text.split('.')
    words = text.split()
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': len(text) / len(sentences) if sentences else 0
    }


def categorize_by_keywords(text: str, topic_config: dict) -> list:
    """Categorize text by topic based on keywords."""
    text_lower = text.lower()
    topics = []
    
    for topic, info in topic_config.items():
        keywords = info['keywords']
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics if topics else ['other']


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Analyze the extracted dataset."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    interim_dir = Path(config_data['paths']['interim_data'])
    logs_dir = Path(config_data['paths']['logs_dir'])
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = interim_dir / 'extracted.jsonl'
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    logger.info("Analyzing dataset...")
    
    # Collect statistics
    languages = Counter()
    doc_lengths = []
    word_counts = []
    sentence_counts = []
    topic_counts = Counter()
    urls_seen = set()
    duplicates = 0
    
    topic_config = config_data['topics']['buckets']
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Analyzing documents"):
            try:
                doc = json.loads(line)
                text = doc.get('text', '')
                url = doc.get('url', '')
                
                # Check for URL duplicates
                if url in urls_seen:
                    duplicates += 1
                    continue
                urls_seen.add(url)
                
                # Language detection
                lang = detect_language(text)
                languages[lang] += 1
                
                # Text statistics
                stats = analyze_text_stats(text)
                doc_lengths.append(stats['char_count'])
                word_counts.append(stats['word_count'])
                sentence_counts.append(stats['sentence_count'])
                
                # Topic categorization
                topics = categorize_by_keywords(text, topic_config)
                for topic in topics:
                    topic_counts[topic] += 1
                
            except Exception as e:
                logger.error(f"Error processing line: {e}")
    
    # Generate analysis report
    total_docs = len(urls_seen)
    
    report = {
        'total_documents': total_docs,
        'duplicates_found': duplicates,
        'languages': dict(languages),
        'english_ratio': languages.get('en', 0) / total_docs if total_docs > 0 else 0,
        'length_stats': {
            'mean_chars': float(np.mean(doc_lengths)) if doc_lengths else 0,
            'median_chars': float(np.median(doc_lengths)) if doc_lengths else 0,
            'min_chars': int(np.min(doc_lengths)) if doc_lengths else 0,
            'max_chars': int(np.max(doc_lengths)) if doc_lengths else 0
        },
        'word_count_stats': {
            'mean': float(np.mean(word_counts)) if word_counts else 0,
            'median': float(np.median(word_counts)) if word_counts else 0,
            'min': int(np.min(word_counts)) if word_counts else 0,
            'max': int(np.max(word_counts)) if word_counts else 0
        },
        'topic_distribution': dict(topic_counts)
    }
    
    # Save report
    report_file = logs_dir / 'analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Analysis complete. Report saved to {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total documents: {total_docs}")
    print(f"Duplicates: {duplicates}")
    print(f"English ratio: {report['english_ratio']:.2%}")
    length_mean = report['length_stats'].get('mean_chars', report['length_stats'].get('mean', 0))
    word_mean = report['word_count_stats'].get('mean', 0)
    length_std = float(np.std(doc_lengths)) if doc_lengths else 0
    word_std = float(np.std(word_counts)) if word_counts else 0
    print(f"\nDocument length (chars): {length_mean:.0f} +/- {length_std:.0f}")
    print(f"Word count: {word_mean:.0f} +/- {word_std:.0f}")
    print(f"\nTop languages:")
    if total_docs > 0:
        for lang, count in languages.most_common(5):
            print(f"  {lang}: {count} ({count/total_docs:.1%})")
    else:
        print("  No documents processed.")
    print(f"\nTopic distribution:")
    if topic_counts:
        for topic, count in topic_counts.most_common():
            print(f"  {topic}: {count}")
    else:
        print("  No topics detected.")
    
    # Generate visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Document length distribution
    axes[0, 0].hist(doc_lengths, bins=50, edgecolor='black')
    axes[0, 0].set_title('Document Length Distribution')
    axes[0, 0].set_xlabel('Characters')
    axes[0, 0].set_ylabel('Frequency')
    
    # Word count distribution
    axes[0, 1].hist(word_counts, bins=50, edgecolor='black', color='green')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].set_xlabel('Words')
    axes[0, 1].set_ylabel('Frequency')
    
    # Language distribution
    langs = list(languages.keys())[:10]
    counts = [languages[l] for l in langs]
    axes[1, 0].bar(langs, counts)
    axes[1, 0].set_title('Language Distribution')
    axes[1, 0].set_xlabel('Language')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Topic distribution
    topics = list(topic_counts.keys())
    topic_vals = [topic_counts[t] for t in topics]
    axes[1, 1].barh(topics, topic_vals)
    axes[1, 1].set_title('Topic Distribution')
    axes[1, 1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig(logs_dir / 'analysis_plots.png', dpi=150)
    logger.info(f"Plots saved to {logs_dir / 'analysis_plots.png'}")
    
    print("="*60)


if __name__ == '__main__':
    main()

