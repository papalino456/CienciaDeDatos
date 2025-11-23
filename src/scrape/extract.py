#!/usr/bin/env python3
"""
Extract main text content from HTML files using trafilatura.
"""

import json
import logging
from pathlib import Path

import click
import trafilatura
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text_from_html(html_path: Path, metadata_path: Path) -> dict | None:
    """Extract main text from HTML file."""
    try:
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        # Extract text using trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )
        
        if text and len(text.strip()) > 100:  # Minimum text length
            # Try to extract title
            title = trafilatura.extract(html, output_format='json')
            if title:
                title_dict = json.loads(title)
                title = title_dict.get('title', '')
            else:
                title = ''
            
            return {
                'url': metadata['url'],
                'title': title,
                'text': text.strip(),
                'timestamp': metadata['timestamp']
            }
        else:
            logger.debug(f"Insufficient text extracted from {metadata['url']}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting text from {html_path}: {e}")
        return None


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Extract text from raw HTML files."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    raw_dir = Path(config_data['paths']['raw_data'])
    interim_dir = Path(config_data['paths']['interim_data'])
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Extracting text from HTML files...")
    
    # Find all HTML files
    html_files = list(raw_dir.glob('*.html'))
    logger.info(f"Found {len(html_files)} HTML files")
    
    extracted_count = 0
    output_file = interim_dir / 'extracted.jsonl'
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for html_path in tqdm(html_files, desc="Extracting text"):
            metadata_path = html_path.with_suffix('.json')
            
            if not metadata_path.exists():
                logger.warning(f"Missing metadata for {html_path}")
                continue
            
            result = extract_text_from_html(html_path, metadata_path)
            
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                extracted_count += 1
    
    # Also process arXiv JSON files (already have text)
    arxiv_files = list(raw_dir.glob('arxiv_*.json'))
    logger.info(f"Found {len(arxiv_files)} arXiv files")
    
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for arxiv_path in tqdm(arxiv_files, desc="Processing arXiv"):
            try:
                with open(arxiv_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get('text') and len(data['text']) > 50:
                    out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    extracted_count += 1
            except Exception as e:
                logger.error(f"Error processing {arxiv_path}: {e}")
    
    logger.info(f"Extraction complete. Extracted {extracted_count} documents.")
    logger.info(f"Output saved to {output_file}")


if __name__ == '__main__':
    main()

