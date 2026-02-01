"""
News Sentiment Data Source для ML-Framework ML Sentiment Engine

Enterprise-grade news aggregation с Context7 patterns и multi-source support.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import re
from urllib.parse import urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from ..utils.logger import get_logger
from ..utils.config import get_config, get_crypto_symbols, get_crypto_keywords
from ..utils.validators import TextContent, CryptoSymbol, validate_text_content, sanitize_text

logger = get_logger(__name__)


class NewsRateLimiter:
    """Rate limiter для news источников"""
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # {domain: [timestamps]}
        self.min_interval = 60.0 / requests_per_minute
    
    async def acquire(self, domain: str):
        """Получение разрешения на запрос для домена"""
        now = time.time()
        
        if domain not in self.requests:
            self.requests[domain] = []
        
        # Очистка старых запросов
        cutoff_time = now - 60
        self.requests[domain] = [req_time for req_time in self.requests[domain] if req_time > cutoff_time]
        
        # Проверка лимита
        if len(self.requests[domain]) >= self.requests_per_minute:
            sleep_time = 60 - (now - min(self.requests[domain])) + 1
            logger.debug(
                "News rate limit reached for domain",
                domain=domain,
                sleep_time=sleep_time
            )
            await asyncio.sleep(sleep_time)
        
        self.requests[domain].append(now)


class NewsArticleExtractor:
    """Извлечение текста из новостных статей"""
    
    @staticmethod
    def extract_content(html: str, url: str) -> Dict[str, str]:
        """
        Извлечение контента из HTML статьи
        
        Args:
            html: HTML содержимое
            url: URL статьи
            
        Returns:
            Dict[str, str]: Извлеченный контент
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Удаление скриптов и стилей
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Поиск основного контента
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '.main-content'
            ]
            
            main_content = None
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            if not main_content:
                return {"title": "", "text": "", "summary": ""}
            
            # Извлечение заголовка
            title = ""
            title_tags = soup.find_all(['h1', 'h2'])
            if title_tags:
                title = title_tags[0].get_text(strip=True)
            
            if not title:
                title_tag = soup.find('title')
                if title_tag:
                    title = title_tag.get_text(strip=True)
            
            # Извлечение основного текста
            paragraphs = main_content.find_all(['p', 'div'])
            text_parts = []
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 50:  # Исключаем короткие фрагменты
                    text_parts.append(text)
            
            full_text = "\n".join(text_parts)
            
            # Создание краткого описания (первые 2-3 предложения)
            sentences = re.split(r'[.!?]+', full_text)
            summary = ". ".join(sentences[:3]).strip()
            if summary and not summary.endswith('.'):
                summary += '.'
            
            return {
                "title": title,
                "text": full_text,
                "summary": summary
            }
            
        except Exception as e:
            logger.error("Error extracting article content", error=e, url=url)
            return {"title": "", "text": "", "summary": ""}


class NewsSentimentSource:
    """
    Enterprise-grade news data source для sentiment analysis
    
    Features:
    - Multi-source RSS feed aggregation
    - Full article content extraction
    - Crypto-focused news filtering
    - Rate limiting per domain
    - Content deduplication
    - Article ranking by relevance
    """
    
    def __init__(self):
        """Инициализация News source"""
        config = get_config()
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limiter = NewsRateLimiter()
        
        # Content extractor
        self.extractor = NewsArticleExtractor()
        
        # Crypto symbols and keywords
        self.crypto_symbols = set(get_crypto_symbols())
        self.crypto_keywords = set(get_crypto_keywords())
        
        # Crypto news sources
        self.news_sources = {
            "crypto_specialized": [
                "https://cointelegraph.com/rss",
                "https://coindesk.com/arc/outboundfeeds/rss/",
                "https://cryptonews.com/feed/",
                "https://bitcoinist.com/feed/",
                "https://cryptopotato.com/feed/",
                "https://ambcrypto.com/feed/",
                "https://cryptoslate.com/feed/",
                "https://decrypt.co/feed",
                "https://www.theblock.co/rss.xml",
                "https://blockworks.co/feed/"
            ],
            "business_mainstream": [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://feeds.reuters.com/reuters/businessNews",
                "https://www.ft.com/technology?format=rss",
                "https://www.cnbc.com/id/10000664/device/rss/rss.html",  # Tech
                "https://www.marketwatch.com/rss/topstories"
            ],
            "tech_focused": [
                "https://techcrunch.com/feed/",
                "https://www.wired.com/feed/rss",
                "https://arstechnica.com/feeds/rss/",
                "https://www.theverge.com/rss/index.xml"
            ]
        }
        
        # Performance metrics
        self.articles_processed = 0
        self.feeds_processed = 0
        self.api_calls_made = 0
        self.last_error = None
        
        # Content deduplication
        self.seen_articles = set()
    
    async def initialize(self):
        """Инициализация HTTP session"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=10)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'ML-Framework-SentimentEngine/1.0 (Crypto Trading Bot)'
            }
        )
        
        logger.info("News source initialized")
    
    async def cleanup(self):
        """Очистка ресурсов"""
        if self.session:
            await self.session.close()
        logger.info("News source cleaned up")
    
    def _extract_crypto_mentions(self, text: str) -> Set[str]:
        """
        Извлечение упоминаний криптовалют из текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            Set[str]: Найденные символы криптовалют
        """
        mentioned_symbols = set()
        text_upper = text.upper()
        
        for symbol in self.crypto_symbols:
            patterns = [
                rf'\b{symbol}\b',
                rf'\${symbol}\b',
                rf'{symbol}/USD',
                rf'{symbol}USD',
                rf'{symbol}-USD'
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    mentioned_symbols.add(symbol)
                    break
        
        return mentioned_symbols
    
    def _is_crypto_relevant(self, text: str, title: str = "") -> bool:
        """
        Проверка релевантности статьи для крипто-анализа
        
        Args:
            text: Текст статьи
            title: Заголовок статьи
            
        Returns:
            bool: True если статья релевантна
        """
        full_text = f"{title} {text}".lower()
        
        # Проверка ключевых слов
        for keyword in self.crypto_keywords:
            if keyword in full_text:
                return True
        
        # Проверка символов криптовалют
        if self._extract_crypto_mentions(full_text):
            return True
        
        # Дополнительные крипто-паттерны
        crypto_patterns = [
            r'\bcryptocurrency\b', r'\bdigital\s+asset', r'\bdigital\s+currency\b',
            r'\bmarket\s+cap\b', r'\btrading\s+volume\b', r'\ball\s*time\s+high\b',
            r'\bbull\s+run\b', r'\bbear\s+market\b', r'\bwhale\s+alert\b'
        ]
        
        for pattern in crypto_patterns:
            if re.search(pattern, full_text):
                return True
        
        return False
    
    def _calculate_relevance_score(self, title: str, text: str, source_url: str) -> float:
        """
        Расчет релевантности статьи
        
        Args:
            title: Заголовок статьи
            text: Текст статьи
            source_url: URL источника
            
        Returns:
            float: Оценка релевантности от 0 до 1
        """
        score = 0.0
        full_text = f"{title} {text}".lower()
        
        # Бонус за специализированные крипто-источники
        domain = urlparse(source_url).netloc.lower()
        crypto_domains = ['cointelegraph', 'coindesk', 'cryptonews', 'bitcoinist', 'cryptopotato']
        if any(cd in domain for cd in crypto_domains):
            score += 0.3
        
        # Количество упоминаний криптовалют
        crypto_mentions = len(self._extract_crypto_mentions(full_text))
        score += min(crypto_mentions * 0.1, 0.3)
        
        # Ключевые слова в заголовке (важнее)
        title_lower = title.lower()
        title_keywords = sum(1 for keyword in self.crypto_keywords if keyword in title_lower)
        score += min(title_keywords * 0.15, 0.4)
        
        # Ключевые слова в тексте
        text_keywords = sum(1 for keyword in self.crypto_keywords if keyword in full_text)
        score += min(text_keywords * 0.02, 0.2)
        
        # Длина статьи (предпочтение более подробным)
        text_length = len(text.split())
        if text_length > 200:
            score += 0.1
        elif text_length > 500:
            score += 0.2
        
        return min(score, 1.0)
    
    async def _fetch_rss_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """
        Получение RSS feed
        
        Args:
            feed_url: URL RSS feed
            
        Returns:
            List[Dict[str, Any]]: Список статей из feed
        """
        try:
            domain = urlparse(feed_url).netloc
            await self.rate_limiter.acquire(domain)
            
            start_time = time.time()
            
            async with self.session.get(feed_url) as response:
                if response.status != 200:
                    logger.warning(f"RSS feed returned status {response.status}", url=feed_url)
                    return []
                
                content = await response.text()
            
            # Парсинг RSS feed (в отдельном executor для блокирующих операций)
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, content)
            
            execution_time = (time.time() - start_time) * 1000
            self.api_calls_made += 1
            
            articles = []
            
            for entry in feed.entries:
                # Проверка на дубликаты
                article_id = entry.get('id', entry.get('link', ''))
                if article_id in self.seen_articles:
                    continue
                
                self.seen_articles.add(article_id)
                
                # Базовая информация о статье
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                link = entry.get('link', '')
                published = entry.get('published_parsed')
                
                if published:
                    pub_date = datetime(*published[:6])
                else:
                    pub_date = datetime.utcnow()
                
                article = {
                    "id": article_id,
                    "title": title,
                    "summary": summary,
                    "link": link,
                    "published_at": pub_date.isoformat(),
                    "source_feed": feed_url,
                    "source_domain": domain
                }
                
                articles.append(article)
            
            logger.debug(
                "RSS feed fetched",
                url=feed_url,
                articles_count=len(articles),
                execution_time_ms=execution_time
            )
            
            return articles
            
        except Exception as e:
            self.last_error = e
            logger.error("Error fetching RSS feed", error=e, url=feed_url)
            return []
    
    async def _fetch_full_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Получение полного текста статьи
        
        Args:
            article: Базовая информация о статье
            
        Returns:
            Optional[Dict[str, Any]]: Обработанная статья с полным текстом
        """
        try:
            link = article.get('link')
            if not link:
                return None
            
            domain = urlparse(link).netloc
            await self.rate_limiter.acquire(domain)
            
            async with self.session.get(link) as response:
                if response.status != 200:
                    logger.debug(f"Article returned status {response.status}", url=link)
                    return None
                
                html = await response.text()
            
            # Извлечение контента
            content = self.extractor.extract_content(html, link)
            
            if not content['text'] or len(content['text']) < 100:
                return None
            
            # Проверка релевантности
            if not self._is_crypto_relevant(content['text'], content['title']):
                return None
            
            # Очистка текста
            cleaned_text = sanitize_text(content['text'])
            if not validate_text_content(cleaned_text, "news"):
                return None
            
            # Расчет релевантности
            relevance_score = self._calculate_relevance_score(
                content['title'],
                content['text'],
                article['source_feed']
            )
            
            processed_article = {
                "id": article['id'],
                "text": cleaned_text,
                "original_text": content['text'],
                "title": content['title'] or article['title'],
                "summary": content['summary'] or article['summary'],
                "symbols_mentioned": list(self._extract_crypto_mentions(content['text'])),
                "source": "news",
                "url": link,
                "published_at": article['published_at'],
                "source_domain": article['source_domain'],
                "source_feed": article['source_feed'],
                "metrics": {
                    "relevance_score": relevance_score,
                    "word_count": len(content['text'].split()),
                    "reading_time": len(content['text'].split()) // 200  # words per minute
                },
                "metadata": {
                    "language": "en",
                    "platform": "news",
                    "content_type": "article",
                    "source_type": self._classify_source(article['source_domain'])
                }
            }
            
            self.articles_processed += 1
            
            return processed_article
            
        except Exception as e:
            logger.error("Error fetching full article", error=e, url=article.get('link'))
            return None
    
    def _classify_source(self, domain: str) -> str:
        """
        Классификация типа источника
        
        Args:
            domain: Домен источника
            
        Returns:
            str: Тип источника
        """
        domain_lower = domain.lower()
        
        if any(cd in domain_lower for cd in ['cointelegraph', 'coindesk', 'cryptonews', 'bitcoinist']):
            return "crypto_specialized"
        elif any(cd in domain_lower for cd in ['bloomberg', 'reuters', 'cnbc', 'marketwatch']):
            return "business_mainstream"
        elif any(cd in domain_lower for cd in ['techcrunch', 'wired', 'ars', 'theverge']):
            return "tech_focused"
        else:
            return "general"
    
    async def fetch_latest_news(
        self,
        source_types: List[str] = None,
        limit: int = 100,
        hours_back: int = 24,
        min_relevance_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Получение последних новостей
        
        Args:
            source_types: Типы источников для поиска
            limit: Максимальное количество статей
            hours_back: Период поиска в часах
            min_relevance_score: Минимальная релевантность
            
        Returns:
            List[Dict[str, Any]]: Список обработанных статей
        """
        if not self.session:
            await self.initialize()
        
        if not source_types:
            source_types = list(self.news_sources.keys())
        
        all_articles = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Получение статей из всех источников
        for source_type in source_types:
            feeds = self.news_sources.get(source_type, [])
            
            for feed_url in feeds:
                try:
                    # Получение RSS feed
                    rss_articles = await self._fetch_rss_feed(feed_url)
                    
                    # Фильтрация по времени
                    recent_articles = [
                        article for article in rss_articles
                        if datetime.fromisoformat(article['published_at'].replace('Z', '+00:00')).replace(tzinfo=None) > cutoff_time
                    ]
                    
                    # Получение полного контента (с ограничением конкурентности)
                    semaphore = asyncio.Semaphore(5)
                    
                    async def process_article(article):
                        async with semaphore:
                            return await self._fetch_full_article(article)
                    
                    tasks = [process_article(article) for article in recent_articles[:20]]  # Лимит на feed
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Фильтрация результатов
                    for result in results:
                        if isinstance(result, dict) and result.get('metrics', {}).get('relevance_score', 0) >= min_relevance_score:
                            all_articles.append(result)
                    
                    self.feeds_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing feed {feed_url}", error=e)
                    continue
        
        # Сортировка по релевантности и времени
        all_articles.sort(
            key=lambda a: (
                a.get('metrics', {}).get('relevance_score', 0),
                a.get('published_at', '')
            ),
            reverse=True
        )
        
        result = all_articles[:limit]
        
        logger.info(
            "Latest news fetched",
            source_types=source_types,
            feeds_processed=self.feeds_processed,
            articles_found=len(result),
            min_relevance=min_relevance_score
        )
        
        return result
    
    async def search_news(
        self,
        keywords: List[str],
        limit: int = 50,
        hours_back: int = 168  # 1 week
    ) -> List[Dict[str, Any]]:
        """
        Поиск новостей по ключевым словам
        
        Args:
            keywords: Ключевые слова для поиска
            limit: Максимальное количество результатов
            hours_back: Период поиска в часах
            
        Returns:
            List[Dict[str, Any]]: Найденные статьи
        """
        # Получение всех новостей
        all_news = await self.fetch_latest_news(
            limit=limit * 2,  # Получаем больше для фильтрации
            hours_back=hours_back,
            min_relevance_score=0.1  # Низкий порог для поиска
        )
        
        # Фильтрация по ключевым словам
        matching_articles = []
        keywords_lower = [k.lower() for k in keywords]
        
        for article in all_news:
            text = f"{article.get('title', '')} {article.get('text', '')}".lower()
            
            if any(keyword in text for keyword in keywords_lower):
                # Расчет релевантности для ключевых слов
                keyword_score = sum(1 for keyword in keywords_lower if keyword in text) / len(keywords_lower)
                article['metrics']['keyword_relevance'] = keyword_score
                matching_articles.append(article)
        
        # Сортировка по релевантности ключевых слов
        matching_articles.sort(
            key=lambda a: a.get('metrics', {}).get('keyword_relevance', 0),
            reverse=True
        )
        
        result = matching_articles[:limit]
        
        logger.info(
            "News search completed",
            keywords=keywords,
            articles_found=len(result),
            total_searched=len(all_news)
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики источника
        
        Returns:
            Dict[str, Any]: Статистика работы
        """
        return {
            "source": "news",
            "articles_processed": self.articles_processed,
            "feeds_processed": self.feeds_processed,
            "api_calls_made": self.api_calls_made,
            "last_error": str(self.last_error) if self.last_error else None,
            "news_sources_count": sum(len(feeds) for feeds in self.news_sources.values()),
            "crypto_symbols_tracked": len(self.crypto_symbols),
            "crypto_keywords_tracked": len(self.crypto_keywords),
            "initialized": self.session is not None,
            "seen_articles_count": len(self.seen_articles),
            "source_categories": list(self.news_sources.keys())
        }


async def create_news_source() -> NewsSentimentSource:
    """
    Factory function для создания News source
    
    Returns:
        NewsSentimentSource: Настроенный источник данных
    """
    source = NewsSentimentSource()
    await source.initialize()
    return source