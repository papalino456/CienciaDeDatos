#!/usr/bin/env python3
"""
Web crawler for mechatronics domain text collection.
Respects robots.txt, implements rate limiting, and saves raw HTML.
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import click
import httpx
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoliteCrawler:
    """Async web crawler with politeness features."""
    
    def __init__(self, config: dict, sources_config: dict):
        self.config = config
        self.sources_config = sources_config
        self.user_agent = config['scrape']['user_agent']
        self.timeout = config['scrape']['timeout']
        self.max_retries = config['scrape']['max_retries']
        self.rate_limit = config['scrape']['request_rate_per_host']
        self.max_pages_per_host = config['scrape']['max_pages_per_host']
        
        self.allowed_hosts = set(sources_config['allowed_hosts'])
        self.disallowed_patterns = sources_config['disallowed_patterns']
        
        self.visited_urls: Set[str] = set()
        self.domain_last_request: Dict[str, float] = {}
        self.domain_page_count: Dict[str, int] = {}
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        
        self.output_dir = Path(config['paths']['raw_data'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _url_hash(self, url: str) -> str:
        """Generate a short hash for URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is allowed by our rules."""
        parsed = urlparse(url)
        
        # Check host allowlist
        if parsed.netloc not in self.allowed_hosts:
            return False
        
        # Check disallowed patterns
        for pattern in self.disallowed_patterns:
            if pattern in url:
                return False
        
        return True
    
    def _can_fetch(self, url: str) -> bool:
        """Check robots.txt permissions."""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if domain not in self.robots_parsers:
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception as e:
                logger.warning(f"Could not read robots.txt for {domain}: {e}")
                # Default to allowing if robots.txt can't be read
                rp = None
            self.robots_parsers[domain] = rp
        
        rp = self.robots_parsers[domain]
        if rp is None:
            return True
        
        return rp.can_fetch(self.user_agent, url)
    
    async def _wait_for_rate_limit(self, domain: str):
        """Implement per-domain rate limiting."""
        if domain in self.domain_last_request:
            elapsed = time.time() - self.domain_last_request[domain]
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
        
        self.domain_last_request[domain] = time.time()
    
    async def _fetch_url(self, client: httpx.AsyncClient, url: str) -> tuple[str, str] | None:
        """Fetch a single URL with retries."""
        domain = urlparse(url).netloc
        
        # Check page limit per domain
        if self.domain_page_count.get(domain, 0) >= self.max_pages_per_host:
            logger.debug(f"Domain page limit reached for {domain}")
            return None
        
        # Rate limiting
        await self._wait_for_rate_limit(domain)
        
        for attempt in range(self.max_retries):
            try:
                response = await client.get(url, timeout=self.timeout, follow_redirects=True)
                if response.status_code == 200:
                    self.domain_page_count[domain] = self.domain_page_count.get(domain, 0) + 1
                    return url, response.text
                else:
                    logger.warning(f"Got status {response.status_code} for {url}")
                    return None
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            absolute_url = urljoin(base_url, href)
            
            # Remove fragments
            absolute_url = absolute_url.split('#')[0]
            
            if self._is_allowed_url(absolute_url) and self._can_fetch(absolute_url):
                links.append(absolute_url)
        
        return links
    
    async def crawl(self):
        """Main crawling loop."""
        seeds = [item['url'] for item in self.sources_config['seed_urls']]
        seed_depths = {item['url']: item['max_depth'] for item in self.sources_config['seed_urls']}
        
        # Queue: (url, depth)
        queue = [(url, 0) for url in seeds]
        
        headers = {'User-Agent': self.user_agent}
        
        async with httpx.AsyncClient(headers=headers) as client:
            pbar = tqdm(desc="Crawling pages", unit="page")
            
            while queue:
                url, depth = queue.pop(0)
                
                if url in self.visited_urls:
                    continue
                
                self.visited_urls.add(url)
                
                result = await self._fetch_url(client, url)
                if result is None:
                    continue
                
                fetched_url, html = result
                
                # Save HTML
                url_hash = self._url_hash(fetched_url)
                output_file = self.output_dir / f"{url_hash}.html"
                
                metadata = {
                    'url': fetched_url,
                    'timestamp': time.time(),
                    'depth': depth
                }
                
                with open(output_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                
                pbar.update(1)
                pbar.set_postfix({'total': len(self.visited_urls), 'queue': len(queue)})
                
                # Extract and queue links if within depth limit
                max_depth = seed_depths.get(url, 1)
                if depth < max_depth:
                    links = self._extract_links(html, fetched_url)
                    for link in links:
                        if link not in self.visited_urls:
                            queue.append((link, depth + 1))
            
            pbar.close()
        
        logger.info(f"Crawling complete. Visited {len(self.visited_urls)} pages.")


async def fetch_arxiv_abstracts(config: dict):
    """Fetch arXiv abstracts via API."""
    import xml.etree.ElementTree as ET
    
    categories = config['scrape']['arxiv_categories']
    max_results = config['scrape']['arxiv_max_results']
    output_dir = Path(config['paths']['raw_data'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://export.arxiv.org/api/query"
    
    async with httpx.AsyncClient(timeout=30) as client:
        for category in categories:
            logger.info(f"Fetching arXiv abstracts for category: {category}")
            
            params = {
                'search_query': f'cat:{category}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = await client.get(base_url, params=params)
                if response.status_code == 200:
                    # Parse XML
                    root = ET.fromstring(response.text)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    
                    entries = root.findall('atom:entry', ns)
                    logger.info(f"Found {len(entries)} entries for {category}")
                    
                    for entry in entries:
                        arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                        title = entry.find('atom:title', ns).text.strip()
                        abstract = entry.find('atom:summary', ns).text.strip()
                        
                        # Save
                        output_file = output_dir / f"arxiv_{arxiv_id.replace('/', '_')}.json"
                        data = {
                            'url': f"https://arxiv.org/abs/{arxiv_id}",
                            'title': title,
                            'text': abstract,
                            'source': 'arxiv',
                            'category': category,
                            'timestamp': time.time()
                        }
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Rate limit for arXiv API
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error fetching arXiv {category}: {e}")


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True, help='Pipeline config file')
def main(config):
    """Run the web crawler."""
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    sources_file = Path('src/scrape/sources.yaml')
    with open(sources_file, 'r') as f:
        sources_config = yaml.safe_load(f)
    
    logger.info("Starting web crawler...")
    
    # Crawl web pages
    crawler = PoliteCrawler(config_data, sources_config)
    asyncio.run(crawler.crawl())
    
    # Fetch arXiv abstracts
    logger.info("Fetching arXiv abstracts...")
    asyncio.run(fetch_arxiv_abstracts(config_data))
    
    logger.info("Crawling complete!")


if __name__ == '__main__':
    main()

