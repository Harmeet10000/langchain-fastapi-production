"""Intelligent web scraping with advanced content extraction and analysis."""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime
import hashlib

import httpx
from bs4 import BeautifulSoup, Comment
from playwright.async_api import async_playwright, Page
from readability import Readability
import trafilatura
from newspaper import Article
import feedparser
from sitemap_parser import SiteMapParser

from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import cache_manager
from src.services.langchain.gemini_service import gemini_service

logger = LoggerAdapter(__name__)


class WebIntelligence:
    """Advanced web scraping with intelligent content extraction."""
    
    def __init__(self):
        """Initialize web intelligence service."""
        self.browser = None
        self.playwright = None
        self.session = None
        
    async def initialize(self):
        """Initialize browser and session."""
        if not self.playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=settings.crawl4ai_headless,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
        
        if not self.session:
            self.session = httpx.AsyncClient(
                timeout=30,
                headers={"User-Agent": settings.crawl4ai_user_agent},
                follow_redirects=True
            )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        if self.session:
            await self.session.aclose()
    
    async def intelligent_scrape(
        self,
        url: str,
        extract_method: str = "auto",  # auto, readability, trafilatura, newspaper, custom
        javascript_render: bool = False,
        extract_metadata: bool = True,
        extract_links: bool = True,
        extract_images: bool = True,
        extract_videos: bool = True,
        extract_structured_data: bool = True,
        extract_social_media: bool = True,
        extract_comments: bool = False,
        analyze_content: bool = True,
        screenshot: bool = False
    ) -> Dict[str, Any]:
        """Intelligently scrape and extract content from a webpage."""
        try:
            result = {
                "url": url,
                "domain": urlparse(url).netloc,
                "timestamp": datetime.utcnow().isoformat(),
                "content": {},
                "metadata": {},
                "links": [],
                "images": [],
                "videos": [],
                "structured_data": {},
                "social_media": {},
                "analysis": {},
                "errors": []
            }
            
            # Get page content
            if javascript_render:
                html, page_data = await self._get_page_with_js(url, screenshot)
                result.update(page_data)
            else:
                html = await self._get_page_static(url)
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract content using specified method
            if extract_method == "auto":
                result["content"] = await self._auto_extract_content(url, html, soup)
            elif extract_method == "readability":
                result["content"] = self._extract_with_readability(html, url)
            elif extract_method == "trafilatura":
                result["content"] = self._extract_with_trafilatura(html, url)
            elif extract_method == "newspaper":
                result["content"] = await self._extract_with_newspaper(url, html)
            else:
                result["content"] = self._custom_extract(soup)
            
            # Extract metadata
            if extract_metadata:
                result["metadata"] = self._extract_metadata(soup, url)
            
            # Extract links
            if extract_links:
                result["links"] = self._extract_links_advanced(soup, url)
            
            # Extract images
            if extract_images:
                result["images"] = self._extract_images_advanced(soup, url)
            
            # Extract videos
            if extract_videos:
                result["videos"] = self._extract_videos(soup, url)
            
            # Extract structured data (JSON-LD, Microdata, etc.)
            if extract_structured_data:
                result["structured_data"] = self._extract_structured_data(soup)
            
            # Extract social media metadata
            if extract_social_media:
                result["social_media"] = self._extract_social_media(soup)
            
            # Extract comments if requested
            if extract_comments:
                result["comments"] = await self._extract_comments(soup, url)
            
            # Analyze content
            if analyze_content and result["content"].get("text"):
                result["analysis"] = await self._analyze_content(
                    result["content"]["text"],
                    result["metadata"]
                )
            
            logger.info(f"Successfully scraped: {url}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _get_page_with_js(
        self,
        url: str,
        screenshot: bool
    ) -> tuple[str, Dict]:
        """Get page content with JavaScript rendering."""
        await self.initialize()
        
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent=settings.crawl4ai_user_agent
        )
        page = await context.new_page()
        
        page_data = {}
        
        try:
            # Navigate and wait for content
            await page.goto(url, wait_until="networkidle", timeout=settings.crawl4ai_timeout)
            
            # Wait for dynamic content
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(2)  # Additional wait for dynamic content
            
            # Scroll to load lazy content
            await self._scroll_page(page)
            
            # Get final HTML
            html = await page.content()
            
            # Take screenshot if requested
            if screenshot:
                screenshot_data = await page.screenshot(full_page=True)
                page_data["screenshot"] = screenshot_data
            
            # Get page metrics
            page_data["metrics"] = await page.evaluate("""
                () => ({
                    scrollHeight: document.documentElement.scrollHeight,
                    clientHeight: document.documentElement.clientHeight,
                    images: document.images.length,
                    links: document.links.length,
                    scripts: document.scripts.length
                })
            """)
            
            # Extract JavaScript-rendered data
            page_data["js_data"] = await self._extract_js_data(page)
            
            return html, page_data
            
        finally:
            await context.close()
    
    async def _get_page_static(self, url: str) -> str:
        """Get page content without JavaScript."""
        await self.initialize()
        response = await self.session.get(url)
        response.raise_for_status()
        return response.text
    
    async def _scroll_page(self, page: Page):
        """Scroll page to load lazy content."""
        await page.evaluate("""
            async () => {
                const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
                const scrollHeight = document.documentElement.scrollHeight;
                const step = 500;
                
                for (let i = 0; i < scrollHeight; i += step) {
                    window.scrollTo(0, i);
                    await delay(100);
                }
                window.scrollTo(0, scrollHeight);
            }
        """)
    
    async def _extract_js_data(self, page: Page) -> Dict:
        """Extract data from JavaScript context."""
        return await page.evaluate("""
            () => {
                const data = {};
                
                // Try to find React props
                const reactRoot = document.querySelector('#root') || document.querySelector('[data-reactroot]');
                if (reactRoot && reactRoot._reactRootContainer) {
                    try {
                        data.react = true;
                    } catch(e) {}
                }
                
                // Try to find Vue instance
                if (window.Vue || window.__VUE__) {
                    data.vue = true;
                }
                
                // Try to find Angular
                if (window.angular || document.querySelector('[ng-app]')) {
                    data.angular = true;
                }
                
                // Get any window variables that might contain data
                if (window.__INITIAL_STATE__) {
                    data.initialState = window.__INITIAL_STATE__;
                }
                
                if (window.__PRELOADED_STATE__) {
                    data.preloadedState = window.__PRELOADED_STATE__;
                }
                
                return data;
            }
        """)
    
    async def _auto_extract_content(
        self,
        url: str,
        html: str,
        soup: BeautifulSoup
    ) -> Dict[str, Any]:
        """Automatically extract content using multiple methods and pick best."""
        results = []
        
        # Try Readability
        try:
            readability_result = self._extract_with_readability(html, url)
            if readability_result.get("text"):
                results.append(("readability", readability_result))
        except:
            pass
        
        # Try Trafilatura
        try:
            trafilatura_result = self._extract_with_trafilatura(html, url)
            if trafilatura_result.get("text"):
                results.append(("trafilatura", trafilatura_result))
        except:
            pass
        
        # Try Newspaper
        try:
            newspaper_result = await self._extract_with_newspaper(url, html)
            if newspaper_result.get("text"):
                results.append(("newspaper", newspaper_result))
        except:
            pass
        
        # Try custom extraction
        custom_result = self._custom_extract(soup)
        if custom_result.get("text"):
            results.append(("custom", custom_result))
        
        # Pick the best result (longest content with title)
        best_result = {}
        best_length = 0
        best_method = "none"
        
        for method, result in results:
            text_length = len(result.get("text", ""))
            has_title = bool(result.get("title"))
            
            score = text_length + (1000 if has_title else 0)
            
            if score > best_length:
                best_length = score
                best_result = result
                best_method = method
        
        best_result["extraction_method"] = best_method
        return best_result
    
    def _extract_with_readability(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using Readability."""
        doc = Readability(html)
        
        return {
            "title": doc.title(),
            "text": doc.summary(),
            "short_title": doc.short_title(),
            "method": "readability"
        }
    
    def _extract_with_trafilatura(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using Trafilatura."""
        # Extract main content
        text = trafilatura.extract(
            html,
            output_format='txt',
            include_comments=False,
            include_tables=True,
            deduplicate=True
        )
        
        # Extract metadata
        metadata = trafilatura.extract_metadata(html)
        
        result = {
            "text": text or "",
            "method": "trafilatura"
        }
        
        if metadata:
            result.update({
                "title": metadata.title,
                "author": metadata.author,
                "date": metadata.date,
                "description": metadata.description,
                "categories": metadata.categories,
                "tags": metadata.tags
            })
        
        return result
    
    async def _extract_with_newspaper(self, url: str, html: str) -> Dict[str, Any]:
        """Extract content using Newspaper3k."""
        article = Article(url)
        article.set_html(html)
        article.parse()
        
        # NLP processing
        try:
            article.nlp()
        except:
            pass
        
        return {
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "top_image": article.top_image,
            "images": list(article.images),
            "videos": article.movies,
            "keywords": article.keywords,
            "summary": article.summary,
            "method": "newspaper"
        }
    
    def _custom_extract(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Custom content extraction with heuristics."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Remove comments
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Find title
        title = ""
        title_tag = soup.find('h1') or soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
        
        # Find main content area
        main_content = ""
        
        # Look for common content containers
        content_selectors = [
            'main', 'article', '[role="main"]',
            '.content', '#content', '.post-content',
            '.entry-content', '.article-body'
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                main_content = content.get_text(separator='\n', strip=True)
                break
        
        # Fallback to largest text block
        if not main_content:
            paragraphs = soup.find_all('p')
            if paragraphs:
                main_content = '\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
        
        return {
            "title": title,
            "text": main_content,
            "method": "custom"
        }
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract comprehensive metadata."""
        metadata = {
            "url": url,
            "domain": urlparse(url).netloc,
            "meta_tags": {},
            "open_graph": {},
            "twitter_card": {},
            "dublin_core": {},
            "schema_org": {}
        }
        
        # Standard meta tags
        for tag in soup.find_all('meta'):
            name = tag.get('name') or tag.get('property') or tag.get('http-equiv')
            content = tag.get('content')
            
            if name and content:
                # Open Graph
                if name.startswith('og:'):
                    metadata["open_graph"][name[3:]] = content
                # Twitter Card
                elif name.startswith('twitter:'):
                    metadata["twitter_card"][name[8:]] = content
                # Dublin Core
                elif name.startswith('dc.') or name.startswith('DC.'):
                    metadata["dublin_core"][name[3:]] = content
                # Regular meta
                else:
                    metadata["meta_tags"][name] = content
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Canonical URL
        canonical = soup.find('link', {'rel': 'canonical'})
        if canonical:
            metadata["canonical_url"] = canonical.get('href')
        
        # Language
        html_tag = soup.find('html')
        if html_tag:
            metadata["language"] = html_tag.get('lang')
        
        # Author
        author = soup.find('meta', {'name': 'author'})
        if author:
            metadata["author"] = author.get('content')
        
        # Published/Modified dates
        published = soup.find('meta', {'property': 'article:published_time'})
        if published:
            metadata["published_date"] = published.get('content')
        
        modified = soup.find('meta', {'property': 'article:modified_time'})
        if modified:
            metadata["modified_date"] = modified.get('content')
        
        return metadata
    
    def _extract_links_advanced(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract links with classification."""
        links = []
        seen = set()
        
        for tag in soup.find_all('a', href=True):
            href = tag['href']
            absolute_url = urljoin(base_url, href)
            
            if absolute_url in seen:
                continue
            
            seen.add(absolute_url)
            
            # Classify link
            link_type = self._classify_link(absolute_url, tag)
            
            links.append({
                "url": absolute_url,
                "text": tag.get_text().strip(),
                "title": tag.get('title', ''),
                "type": link_type,
                "internal": urlparse(absolute_url).netloc == urlparse(base_url).netloc,
                "attributes": {
                    "rel": tag.get('rel', []),
                    "target": tag.get('target', ''),
                    "class": tag.get('class', [])
                }
            })
        
        return links
    
    def _classify_link(self, url: str, tag) -> str:
        """Classify link type."""
        url_lower = url.lower()
        text_lower = tag.get_text().lower()
        
        # Social media
        social_domains = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com', 'youtube.com']
        if any(domain in url_lower for domain in social_domains):
            return "social"
        
        # Downloads
        download_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar']
        if any(url_lower.endswith(ext) for ext in download_extensions):
            return "download"
        
        # Email
        if url_lower.startswith('mailto:'):
            return "email"
        
        # Phone
        if url_lower.startswith('tel:'):
            return "phone"
        
        # Navigation
        nav_keywords = ['home', 'about', 'contact', 'services', 'products', 'blog', 'news']
        if any(keyword in text_lower for keyword in nav_keywords):
            return "navigation"
        
        # External
        if urlparse(url).netloc != urlparse(tag.get('base_url', '')).netloc:
            return "external"
        
        return "general"
    
    def _extract_images_advanced(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract images with metadata."""
        images = []
        seen = set()
        
        for img in soup.find_all('img'):
            src = img.get('src', img.get('data-src', img.get('data-lazy-src', '')))
            if not src or src in seen:
                continue
            
            absolute_url = urljoin(base_url, src)
            seen.add(absolute_url)
            
            # Get image metadata
            images.append({
                "url": absolute_url,
                "alt": img.get('alt', ''),
                "title": img.get('title', ''),
                "width": img.get('width', ''),
                "height": img.get('height', ''),
                "loading": img.get('loading', ''),
                "srcset": img.get('srcset', ''),
                "sizes": img.get('sizes', ''),
                "class": img.get('class', [])
            })
        
        # Also extract from picture elements
        for picture in soup.find_all('picture'):
            sources = []
            for source in picture.find_all('source'):
                sources.append({
                    "srcset": source.get('srcset', ''),
                    "media": source.get('media', ''),
                    "type": source.get('type', '')
                })
            
            img = picture.find('img')
            if img and img.get('src'):
                images.append({
                    "url": urljoin(base_url, img.get('src')),
                    "alt": img.get('alt', ''),
                    "sources": sources,
                    "responsive": True
                })
        
        return images
    
    def _extract_videos(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract video content."""
        videos = []
        
        # HTML5 videos
        for video in soup.find_all('video'):
            video_data = {
                "type": "html5",
                "src": video.get('src', ''),
                "poster": video.get('poster', ''),
                "width": video.get('width', ''),
                "height": video.get('height', ''),
                "controls": video.has_attr('controls'),
                "autoplay": video.has_attr('autoplay'),
                "sources": []
            }
            
            for source in video.find_all('source'):
                video_data["sources"].append({
                    "src": urljoin(base_url, source.get('src', '')),
                    "type": source.get('type', '')
                })
            
            videos.append(video_data)
        
        # YouTube embeds
        for iframe in soup.find_all('iframe'):
            src = iframe.get('src', '')
            if 'youtube.com/embed' in src or 'youtu.be' in src:
                videos.append({
                    "type": "youtube",
                    "src": src,
                    "width": iframe.get('width', ''),
                    "height": iframe.get('height', '')
                })
            elif 'vimeo.com' in src:
                videos.append({
                    "type": "vimeo",
                    "src": src,
                    "width": iframe.get('width', ''),
                    "height": iframe.get('height', '')
                })
        
        return videos
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict:
        """Extract structured data (JSON-LD, Microdata, RDFa)."""
        structured = {
            "json_ld": [],
            "microdata": [],
            "rdfa": []
        }
        
        # JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured["json_ld"].append(data)
            except:
                pass
        
        # Microdata
        for item in soup.find_all(attrs={'itemscope': True}):
            item_data = {
                "type": item.get('itemtype', ''),
                "properties": {}
            }
            
            for prop in item.find_all(attrs={'itemprop': True}):
                prop_name = prop.get('itemprop')
                prop_value = prop.get('content') or prop.get_text().strip()
                item_data["properties"][prop_name] = prop_value
            
            structured["microdata"].append(item_data)
        
        return structured
    
    def _extract_social_media(self, soup: BeautifulSoup) -> Dict:
        """Extract social media links and metadata."""
        social = {
            "profiles": [],
            "share_counts": {},
            "social_meta": {}
        }
        
        # Common social media domains
        social_domains = {
            'facebook.com': 'facebook',
            'twitter.com': 'twitter',
            'x.com': 'twitter',
            'linkedin.com': 'linkedin',
            'instagram.com': 'instagram',
            'youtube.com': 'youtube',
            'pinterest.com': 'pinterest',
            'tiktok.com': 'tiktok',
            'reddit.com': 'reddit'
        }
        
        # Find social links
        for link in soup.find_all('a', href=True):
            href = link['href']
            for domain, platform in social_domains.items():
                if domain in href:
                    social["profiles"].append({
                        "platform": platform,
                        "url": href,
                        "text": link.get_text().strip()
                    })
                    break
        
        # Open Graph social
        og_tags = soup.find_all('meta', property=re.compile('^og:'))
        for tag in og_tags:
            prop = tag.get('property')
            content = tag.get('content')
            if prop and content:
                social["social_meta"][prop] = content
        
        # Twitter Card
        twitter_tags = soup.find_all('meta', attrs={'name': re.compile('^twitter:')})
        for tag in twitter_tags:
            name = tag.get('name')
            content = tag.get('content')
            if name and content:
                social["social_meta"][name] = content
        
        return social
    
    async def _extract_comments(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Extract comments if visible on page."""
        comments = []
        
        # Common comment selectors
        comment_selectors = [
            '.comment', '.comments',
            '[id*="comment"]', '[class*="comment"]',
            '.review', '.reviews'
        ]
        
        for selector in comment_selectors:
            comment_elements = soup.select(selector)
            for elem in comment_elements:
                comment_data = {
                    "text": elem.get_text().strip(),
                    "author": "",
                    "date": ""
                }
                
                # Try to find author
                author_elem = elem.select_one('[class*="author"], [class*="user"], [class*="name"]')
                if author_elem:
                    comment_data["author"] = author_elem.get_text().strip()
                
                # Try to find date
                date_elem = elem.select_one('[class*="date"], [class*="time"], time')
                if date_elem:
                    comment_data["date"] = date_elem.get_text().strip()
                
                if comment_data["text"]:
                    comments.append(comment_data)
        
        return comments[:50]  # Limit to 50 comments
    
    async def _analyze_content(self, text: str, metadata: Dict) -> Dict:
        """Analyze extracted content."""
        analysis = {
            "statistics": {
                "word_count": len(text.split()),
                "char_count": len(text),
                "sentence_count": len(text.split('.')),
                "paragraph_count": len(text.split('\n\n'))
            },
            "readability": {},
            "language": metadata.get("language", "unknown"),
            "topics": [],
            "sentiment": {}
        }
        
        # Calculate readability scores
        try:
            from textstat import flesch_reading_ease, flesch_kincaid_grade
            analysis["readability"]["flesch_reading_ease"] = flesch_reading_ease(text)
            analysis["readability"]["flesch_kincaid_grade"] = flesch_kincaid_grade(text)
        except:
            pass
        
        # Use Gemini for advanced analysis if available
        if gemini_service and len(text) > 100:
            try:
                prompt = f"""
                Analyze the following text and provide:
                1. Main topics (3-5 keywords)
                2. Overall sentiment (positive/negative/neutral)
                3. Content type (news/blog/product/documentation/etc)
                
                Text: {text[:2000]}
                
                Return as JSON with keys: topics, sentiment, content_type
                """
                
                response = await gemini_service.generate_response(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                
                # Try to parse response as JSON
                try:
                    ai_analysis = json.loads(response)
                    analysis.update(ai_analysis)
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")
        
        return analysis
    
    async def crawl_website(
        self,
        start_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
        allowed_domains: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[Dict]:
        """Crawl an entire website with depth control."""
        visited = set()
        to_visit = [(start_url, 0)]  # (url, depth)
        results = []
        
        if not allowed_domains:
            allowed_domains = [urlparse(start_url).netloc]
        
        if not exclude_patterns:
            exclude_patterns = [
                r'\.pdf$', r'\.zip$', r'\.exe$',
                r'/tag/', r'/category/', r'/page/\d+'
            ]
        
        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            # Check if URL should be excluded
            if any(re.search(pattern, url) for pattern in exclude_patterns):
                continue
            
            # Check if domain is allowed
            if not any(domain in url for domain in allowed_domains):
                continue
            
            visited.add(url)
            
            # Scrape page
            page_data = await self.intelligent_scrape(
                url,
                extract_method="auto",
                analyze_content=False  # Skip analysis for bulk crawling
            )
            
            results.append(page_data)
            
            # Add new links to queue
            if depth < max_depth:
                for link in page_data.get("links", []):
                    link_url = link.get("url")
                    if link_url and link_url not in visited:
                        to_visit.append((link_url, depth + 1))
            
            # Be polite
            await asyncio.sleep(1)
        
        logger.info(f"Crawled {len(results)} pages from {start_url}")
        return results
    
    async def monitor_website(
        self,
        url: str,
        check_interval: int = 3600,  # 1 hour
        changes_callback = None
    ):
        """Monitor a website for changes."""
        last_content_hash = None
        
        while True:
            try:
                # Scrape current content
                result = await self.intelligent_scrape(url)
                current_content = result.get("content", {}).get("text", "")
                
                # Calculate hash
                current_hash = hashlib.sha256(current_content.encode()).hexdigest()
                
                # Check for changes
                if last_content_hash and current_hash != last_content_hash:
                    change_data = {
                        "url": url,
                        "timestamp": datetime.utcnow().isoformat(),
                        "previous_hash": last_content_hash,
                        "current_hash": current_hash,
                        "content": result
                    }
                    
                    logger.info(f"Change detected on {url}")
                    
                    if changes_callback:
                        await changes_callback(change_data)
                
                last_content_hash = current_hash
                
            except Exception as e:
                logger.error(f"Monitoring error for {url}: {e}")
            
            # Wait for next check
            await asyncio.sleep(check_interval)


# Global instance
web_intelligence = WebIntelligence()