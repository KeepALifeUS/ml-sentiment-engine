"""
Reddit Sentiment Data Source для ML-Framework ML Sentiment Engine

Enterprise-grade Reddit data collection с Context7 patterns и async support.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import re

import praw
import aiohttp
from prawcore import ResponseException

from ..utils.logger import get_logger
from ..utils.config import get_config, get_crypto_symbols, get_crypto_keywords
from ..utils.validators import TextContent, CryptoSymbol, validate_text_content, sanitize_text

logger = get_logger(__name__)


class RedditRateLimiter:
    """Rate limiter для Reddit API"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
    
    async def acquire(self):
        """Получение разрешения на запрос"""
        now = time.time()
        
        # Минимальный интервал между запросами
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        # Очистка старых запросов
        cutoff_time = now - 60  # Окно в 1 минуту
        self.requests = [req_time for req_time in self.requests if req_time > cutoff_time]
        
        # Проверка лимита
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - min(self.requests)) + 1
            logger.warning(
                "Reddit rate limit reached, sleeping",
                sleep_time=sleep_time,
                requests_count=len(self.requests)
            )
            await asyncio.sleep(sleep_time)
        
        self.requests.append(time.time())
        self.last_request_time = time.time()


class RedditSentimentSource:
    """
    Enterprise-grade Reddit data source для sentiment analysis
    
    Features:
    - Multi-subreddit monitoring
    - Hot, New, Top posts tracking
    - Comment sentiment analysis
    - Crypto-specific subreddits focus
    - Rate limiting compliance
    - Async execution wrapper
    """
    
    def __init__(self):
        """Инициализация Reddit source"""
        config = get_config()
        
        # Reddit API credentials
        self.client_id = config.social.reddit_client_id
        self.client_secret = config.social.reddit_client_secret
        self.user_agent = config.social.reddit_user_agent
        
        # Reddit client
        self.reddit: Optional[praw.Reddit] = None
        
        # Rate limiting
        self.rate_limiter = RedditRateLimiter()
        
        # Crypto symbols and keywords
        self.crypto_symbols = set(get_crypto_symbols())
        self.crypto_keywords = set(get_crypto_keywords())
        
        # Crypto-specific subreddits
        self.crypto_subreddits = [
            "cryptocurrency",
            "bitcoin",
            "ethereum", 
            "cryptomarkets",
            "altcoin",
            "defi",
            "nft",
            "dogecoin",
            "cardano",
            "solana",
            "polkadot",
            "chainlink",
            "binance",
            "coinbase",
            "crypto",
            "bitcoinmarkets",
            "ethtrader",
            "satoshistreetbets",
            "cryptocurrencies",
            "investing"  # Общий инвест-саб с крипто-контентом
        ]
        
        # Performance metrics
        self.posts_processed = 0
        self.comments_processed = 0
        self.api_calls_made = 0
        self.last_error = None
    
    async def initialize(self):
        """Инициализация Reddit API client"""
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Reddit Client ID and Secret are required")
        
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            requestor_kwargs={"session": aiohttp.ClientSession()}
        )
        
        # Проверка подключения
        try:
            await self._make_reddit_call(lambda: self.reddit.user.me())
            logger.info("Reddit source initialized successfully")
        except Exception as e:
            logger.warning("Reddit authentication failed, using read-only mode", error=e)
    
    async def cleanup(self):
        """Очистка ресурсов"""
        if self.reddit and hasattr(self.reddit._core._requestor, 'session'):
            await self.reddit._core._requestor.session.close()
        logger.info("Reddit source cleaned up")
    
    async def _make_reddit_call(self, reddit_call):
        """
        Выполнение Reddit API call с rate limiting
        
        Args:
            reddit_call: Функция для вызова Reddit API
            
        Returns:
            Any: Результат API call
        """
        await self.rate_limiter.acquire()
        
        try:
            start_time = time.time()
            
            # Выполнение в executor для блокирующих вызовов PRAW
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, reddit_call)
            
            execution_time = (time.time() - start_time) * 1000
            self.api_calls_made += 1
            
            logger.debug(
                "Reddit API call successful",
                execution_time_ms=execution_time,
                total_calls=self.api_calls_made
            )
            
            return result
            
        except Exception as e:
            self.last_error = e
            logger.error("Reddit API call failed", error=e)
            raise
    
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
        
        # Поиск символов криптовалют
        for symbol in self.crypto_symbols:
            patterns = [
                rf'\b{symbol}\b',  # Точное совпадение
                rf'\${symbol}\b',  # С префиксом $
                rf'#{symbol}\b',   # Hashtag
                rf'{symbol}/USD',  # Торговая пара
                rf'{symbol}USD',   # Торговая пара без слеша
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    mentioned_symbols.add(symbol)
                    break
        
        return mentioned_symbols
    
    def _is_crypto_relevant(self, text: str) -> bool:
        """
        Проверка релевантности текста для крипто-анализа
        
        Args:
            text: Текст для проверки
            
        Returns:
            bool: True если текст релевантен
        """
        text_lower = text.lower()
        
        # Проверка ключевых слов
        for keyword in self.crypto_keywords:
            if keyword in text_lower:
                return True
        
        # Проверка символов криптовалют
        if self._extract_crypto_mentions(text):
            return True
        
        # Дополнительные крипто-паттерны
        crypto_patterns = [
            r'\bcrypto\b', r'\bblockchain\b', r'\bhodl\b', r'\btothemoon\b',
            r'\bwhale\b', r'\bpump\b', r'\bdump\b', r'\bfud\b', r'\bfomo\b',
            r'\baltcoin\b', r'\bdefi\b', r'\bnft\b', r'\bweb3\b'
        ]
        
        for pattern in crypto_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    async def fetch_subreddit_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        sort_type: str = "hot",  # hot, new, top
        time_filter: str = "day"  # hour, day, week, month, year, all
    ) -> List[Dict[str, Any]]:
        """
        Получение постов из конкретного subreddit
        
        Args:
            subreddit_name: Название subreddit
            limit: Максимальное количество постов
            sort_type: Тип сортировки
            time_filter: Временной фильтр для top posts
            
        Returns:
            List[Dict[str, Any]]: Список обработанных постов
        """
        try:
            posts = []
            
            async def get_posts():
                subreddit = self.reddit.subreddit(subreddit_name)
                
                if sort_type == "hot":
                    return list(subreddit.hot(limit=limit))
                elif sort_type == "new":
                    return list(subreddit.new(limit=limit))
                elif sort_type == "top":
                    return list(subreddit.top(time_filter=time_filter, limit=limit))
                else:
                    return list(subreddit.hot(limit=limit))
            
            raw_posts = await self._make_reddit_call(get_posts)
            
            for post in raw_posts:
                processed_post = await self._process_post(post, subreddit_name)
                if processed_post and self._is_crypto_relevant(processed_post["text"]):
                    posts.append(processed_post)
            
            self.posts_processed += len(posts)
            
            logger.info(
                "Posts fetched from subreddit",
                subreddit=subreddit_name,
                sort_type=sort_type,
                raw_posts=len(raw_posts),
                relevant_posts=len(posts)
            )
            
            return posts
            
        except Exception as e:
            logger.error(f"Error fetching posts from r/{subreddit_name}", error=e)
            return []
    
    async def fetch_post_comments(
        self,
        post_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Получение комментариев к посту
        
        Args:
            post_id: ID поста Reddit
            limit: Максимальное количество комментариев
            
        Returns:
            List[Dict[str, Any]]: Список обработанных комментариев
        """
        try:
            comments = []
            
            async def get_comments():
                submission = self.reddit.submission(id=post_id)
                submission.comments.replace_more(limit=0)  # Убираем "load more"
                return list(submission.comments.list()[:limit])
            
            raw_comments = await self._make_reddit_call(get_comments)
            
            for comment in raw_comments:
                processed_comment = await self._process_comment(comment, post_id)
                if processed_comment and self._is_crypto_relevant(processed_comment["text"]):
                    comments.append(processed_comment)
            
            self.comments_processed += len(comments)
            
            logger.debug(
                "Comments fetched for post",
                post_id=post_id,
                raw_comments=len(raw_comments),
                relevant_comments=len(comments)
            )
            
            return comments
            
        except Exception as e:
            logger.error(f"Error fetching comments for post {post_id}", error=e)
            return []
    
    async def _process_post(self, post, subreddit_name: str) -> Optional[Dict[str, Any]]:
        """
        Обработка поста Reddit
        
        Args:
            post: Объект поста от Reddit API
            subreddit_name: Название subreddit
            
        Returns:
            Optional[Dict[str, Any]]: Обработанный пост или None
        """
        try:
            # Создание текста из заголовка и содержимого
            title = post.title or ""
            selftext = post.selftext or ""
            full_text = f"{title}. {selftext}".strip()
            
            if not full_text or len(full_text) < 10:
                return None
            
            # Очистка текста
            cleaned_text = sanitize_text(full_text)
            if not cleaned_text:
                return None
            
            # Валидация контента
            if not validate_text_content(cleaned_text, "reddit"):
                return None
            
            # Определение флаира
            flair = post.link_flair_text or ""
            
            processed_post = {
                "id": post.id,
                "text": cleaned_text,
                "original_text": full_text,
                "title": title,
                "symbols_mentioned": list(self._extract_crypto_mentions(full_text)),
                "source": "reddit",
                "subreddit": subreddit_name,
                "created_at": datetime.fromtimestamp(post.created_utc).isoformat(),
                "author": str(post.author) if post.author else "deleted",
                "url": post.url,
                "permalink": f"https://reddit.com{post.permalink}",
                "metrics": {
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "awards": post.total_awards_received
                },
                "metadata": {
                    "language": "en",  # Reddit в основном английский
                    "platform": "reddit",
                    "content_type": "post",
                    "flair": flair,
                    "is_nsfw": post.over_18,
                    "is_spoiler": post.spoiler,
                    "is_stickied": post.stickied,
                    "gilded": post.gilded > 0
                }
            }
            
            return processed_post
            
        except Exception as e:
            logger.error("Error processing Reddit post", error=e, post_id=getattr(post, 'id', 'unknown'))
            return None
    
    async def _process_comment(self, comment, post_id: str) -> Optional[Dict[str, Any]]:
        """
        Обработка комментария Reddit
        
        Args:
            comment: Объект комментария от Reddit API
            post_id: ID родительского поста
            
        Returns:
            Optional[Dict[str, Any]]: Обработанный комментарий или None
        """
        try:
            text = comment.body or ""
            
            if not text or len(text) < 5 or text == "[deleted]" or text == "[removed]":
                return None
            
            # Очистка текста
            cleaned_text = sanitize_text(text)
            if not cleaned_text:
                return None
            
            # Валидация контента
            if not validate_text_content(cleaned_text, "reddit"):
                return None
            
            processed_comment = {
                "id": comment.id,
                "text": cleaned_text,
                "original_text": text,
                "symbols_mentioned": list(self._extract_crypto_mentions(text)),
                "source": "reddit",
                "parent_post_id": post_id,
                "created_at": datetime.fromtimestamp(comment.created_utc).isoformat(),
                "author": str(comment.author) if comment.author else "deleted",
                "permalink": f"https://reddit.com{comment.permalink}",
                "metrics": {
                    "score": comment.score,
                    "is_submitter": comment.is_submitter,
                    "gilded": comment.gilded
                },
                "metadata": {
                    "language": "en",
                    "platform": "reddit",
                    "content_type": "comment",
                    "depth": comment.depth,
                    "is_root": comment.parent_id.startswith("t3_")  # t3_ = link/post
                }
            }
            
            return processed_comment
            
        except Exception as e:
            logger.error("Error processing Reddit comment", error=e, comment_id=getattr(comment, 'id', 'unknown'))
            return None
    
    async def fetch_all_crypto_content(
        self,
        limit_per_subreddit: int = 50,
        include_comments: bool = True,
        max_comments_per_post: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Получение всего крипто-контента из отслеживаемых subreddit
        
        Args:
            limit_per_subreddit: Лимит постов на subreddit
            include_comments: Включать ли комментарии
            max_comments_per_post: Максимум комментариев на пост
            
        Returns:
            List[Dict[str, Any]]: Список всего контента
        """
        all_content = []
        
        for subreddit in self.crypto_subreddits:
            try:
                # Получение постов
                posts = await self.fetch_subreddit_posts(
                    subreddit,
                    limit=limit_per_subreddit,
                    sort_type="hot"
                )
                all_content.extend(posts)
                
                # Получение комментариев для топовых постов
                if include_comments:
                    top_posts = sorted(posts, key=lambda p: p["metrics"]["score"], reverse=True)[:10]
                    
                    for post in top_posts:
                        comments = await self.fetch_post_comments(
                            post["id"],
                            limit=max_comments_per_post
                        )
                        all_content.extend(comments)
                
                # Пауза между subreddit
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing subreddit r/{subreddit}", error=e)
                continue
        
        logger.info(
            "All crypto content fetched",
            subreddits_processed=len(self.crypto_subreddits),
            total_content_items=len(all_content),
            posts=self.posts_processed,
            comments=self.comments_processed
        )
        
        return all_content
    
    async def search_reddit(
        self,
        query: str,
        subreddit: str = "all",
        limit: int = 100,
        sort: str = "relevance",  # relevance, hot, top, new, comments
        time_filter: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Поиск по Reddit
        
        Args:
            query: Поисковый запрос
            subreddit: Subreddit для поиска ("all" для всех)
            limit: Лимит результатов
            sort: Сортировка результатов
            time_filter: Временной фильтр
            
        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        try:
            results = []
            
            async def search():
                target_subreddit = self.reddit.subreddit(subreddit)
                return list(target_subreddit.search(
                    query,
                    sort=sort,
                    time_filter=time_filter,
                    limit=limit
                ))
            
            posts = await self._make_reddit_call(search)
            
            for post in posts:
                processed_post = await self._process_post(post, subreddit)
                if processed_post:
                    results.append(processed_post)
            
            logger.info(
                "Reddit search completed",
                query=query,
                subreddit=subreddit,
                results_found=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error("Reddit search failed", error=e, query=query)
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики источника
        
        Returns:
            Dict[str, Any]: Статистика работы
        """
        return {
            "source": "reddit",
            "posts_processed": self.posts_processed,
            "comments_processed": self.comments_processed,
            "api_calls_made": self.api_calls_made,
            "last_error": str(self.last_error) if self.last_error else None,
            "crypto_subreddits_tracked": len(self.crypto_subreddits),
            "crypto_symbols_tracked": len(self.crypto_symbols),
            "crypto_keywords_tracked": len(self.crypto_keywords),
            "initialized": self.reddit is not None,
            "subreddits": self.crypto_subreddits
        }


async def create_reddit_source() -> RedditSentimentSource:
    """
    Factory function для создания Reddit source
    
    Returns:
        RedditSentimentSource: Настроенный источник данных
    """
    source = RedditSentimentSource()
    await source.initialize()
    return source