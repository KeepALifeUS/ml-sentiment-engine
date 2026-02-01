"""
Twitter/X Sentiment Data Source для ML-Framework ML Sentiment Engine

Enterprise-grade Twitter data collection с Context7 patterns и circuit breaker protection.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import re

import aiohttp
import tweepy
from tweepy.asynchronous import AsyncClient

from ..utils.logger import get_logger
from ..utils.config import get_config, get_crypto_symbols, get_crypto_keywords
from ..utils.validators import TextContent, CryptoSymbol, validate_text_content, sanitize_text

logger = get_logger(__name__)


class TwitterCircuitBreaker:
    """Circuit breaker для Twitter API"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Проверка возможности выполнения запроса"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        """Регистрация успешного запроса"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Регистрация неуспешного запроса"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "Twitter circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


class TwitterRateLimiter:
    """Rate limiter для Twitter API"""
    
    def __init__(self, requests_per_window: int = 300, window_minutes: int = 15):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_minutes * 60
        self.requests = []
    
    async def acquire(self):
        """Получение разрешения на запрос"""
        now = time.time()
        
        # Удаление старых запросов
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window_seconds]
        
        if len(self.requests) >= self.requests_per_window:
            # Ожидание до окончания окна
            oldest_request = min(self.requests)
            sleep_time = self.window_seconds - (now - oldest_request) + 1
            
            logger.warning(
                "Twitter rate limit reached, sleeping",
                sleep_time=sleep_time,
                requests_count=len(self.requests)
            )
            
            await asyncio.sleep(sleep_time)
        
        self.requests.append(now)


class TwitterSentimentSource:
    """
    Enterprise-grade Twitter data source для sentiment analysis
    
    Features:
    - Async streaming support
    - Circuit breaker protection
    - Rate limiting compliance
    - Real-time и historical data
    - Crypto-specific filtering
    - Text preprocessing
    """
    
    def __init__(self):
        """Инициализация Twitter source"""
        config = get_config()
        
        # Twitter API credentials
        self.bearer_token = config.social.twitter_bearer_token
        self.api_key = config.social.twitter_api_key
        self.api_secret = config.social.twitter_api_secret
        self.access_token = config.social.twitter_access_token
        self.access_token_secret = config.social.twitter_access_token_secret
        
        # Async client
        self.client: Optional[AsyncClient] = None
        
        # Protection mechanisms
        self.circuit_breaker = TwitterCircuitBreaker()
        self.rate_limiter = TwitterRateLimiter()
        
        # Crypto symbols and keywords
        self.crypto_symbols = set(get_crypto_symbols())
        self.crypto_keywords = set(get_crypto_keywords())
        
        # Performance metrics
        self.tweets_processed = 0
        self.api_calls_made = 0
        self.last_error = None
    
    async def initialize(self):
        """Инициализация Twitter API client"""
        if not self.bearer_token:
            raise ValueError("Twitter Bearer Token is required")
        
        self.client = AsyncClient(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True
        )
        
        logger.info("Twitter source initialized")
    
    async def cleanup(self):
        """Очистка ресурсов"""
        if self.client:
            await self.client.session.close()
        logger.info("Twitter source cleaned up")
    
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
        
        return False
    
    async def _make_api_call(self, api_call):
        """
        Выполнение API call с protection mechanisms
        
        Args:
            api_call: Функция для вызова API
            
        Returns:
            Any: Результат API call
        """
        if not self.circuit_breaker.can_execute():
            raise Exception("Twitter circuit breaker is open")
        
        await self.rate_limiter.acquire()
        
        try:
            start_time = time.time()
            result = await api_call()
            execution_time = (time.time() - start_time) * 1000
            
            self.circuit_breaker.record_success()
            self.api_calls_made += 1
            
            logger.debug(
                "Twitter API call successful",
                execution_time_ms=execution_time,
                total_calls=self.api_calls_made
            )
            
            return result
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.last_error = e
            
            logger.error(
                "Twitter API call failed",
                error=e,
                api_calls=self.api_calls_made
            )
            raise
    
    async def search_tweets(
        self,
        symbols: List[str] = None,
        limit: int = 100,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Поиск твитов по символам криптовалют
        
        Args:
            symbols: Список символов для поиска
            limit: Максимальное количество твитов
            hours_back: Период поиска в часах
            
        Returns:
            List[Dict[str, Any]]: Список обработанных твитов
        """
        if not self.client:
            await self.initialize()
        
        if not symbols:
            symbols = list(self.crypto_symbols)[:5]  # Топ-5 по умолчанию
        
        all_tweets = []
        
        for symbol in symbols:
            try:
                # Построение поискового запроса
                query_parts = [
                    f"${symbol}",
                    f"#{symbol}",
                    f"{symbol} crypto",
                    f"{symbol} bitcoin"
                ]
                query = " OR ".join(query_parts)
                query += " -is:retweet lang:en"  # Исключаем ретвиты, только английский
                
                # Временные рамки
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours_back)
                
                # API call
                async def api_call():
                    return await self.client.search_recent_tweets(
                        query=query,
                        max_results=min(limit, 100),  # Twitter API limit
                        start_time=start_time,
                        end_time=end_time,
                        tweet_fields=["created_at", "author_id", "public_metrics", "context_annotations"],
                        user_fields=["username", "verified", "public_metrics"]
                    )
                
                response = await self._make_api_call(api_call)
                
                if not response.data:
                    continue
                
                # Обработка полученных твитов
                for tweet in response.data:
                    processed_tweet = await self._process_tweet(tweet, symbol)
                    if processed_tweet and self._is_crypto_relevant(processed_tweet["text"]):
                        all_tweets.append(processed_tweet)
                
                logger.info(
                    "Tweets fetched for symbol",
                    symbol=symbol,
                    tweets_count=len(response.data),
                    relevant_tweets=len([t for t in all_tweets if t.get("symbol") == symbol])
                )
                
                # Пауза между символами
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching tweets for symbol {symbol}", error=e)
                continue
        
        self.tweets_processed += len(all_tweets)
        
        logger.info(
            "Twitter search completed",
            symbols=symbols,
            total_tweets=len(all_tweets),
            processed_total=self.tweets_processed
        )
        
        return all_tweets
    
    async def _process_tweet(self, tweet, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Обработка одного твита
        
        Args:
            tweet: Объект твита от Twitter API
            symbol: Связанный символ криптовалюты
            
        Returns:
            Optional[Dict[str, Any]]: Обработанный твит или None
        """
        try:
            # Извлечение базовой информации
            text = tweet.text
            if not text:
                return None
            
            # Очистка текста
            cleaned_text = sanitize_text(text)
            if not cleaned_text or len(cleaned_text) < 10:
                return None
            
            # Валидация контента
            if not validate_text_content(cleaned_text, "twitter"):
                return None
            
            # Извлечение метрик
            metrics = tweet.public_metrics or {}
            
            # Определение влиятельности
            engagement_score = (
                metrics.get("like_count", 0) * 1 +
                metrics.get("retweet_count", 0) * 2 +
                metrics.get("reply_count", 0) * 1.5 +
                metrics.get("quote_count", 0) * 2
            )
            
            processed_tweet = {
                "id": tweet.id,
                "text": cleaned_text,
                "original_text": text,
                "symbol": symbol,
                "symbols_mentioned": list(self._extract_crypto_mentions(text)),
                "source": "twitter",
                "created_at": tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
                "author_id": tweet.author_id,
                "metrics": {
                    "likes": metrics.get("like_count", 0),
                    "retweets": metrics.get("retweet_count", 0),
                    "replies": metrics.get("reply_count", 0),
                    "quotes": metrics.get("quote_count", 0),
                    "engagement_score": engagement_score
                },
                "metadata": {
                    "language": "en",  # Фильтруем только английские твиты
                    "platform": "twitter",
                    "content_type": "text",
                    "is_verified": False,  # Нужна дополнительная информация о пользователе
                    "follower_count": 0   # Нужна дополнительная информация о пользователе
                }
            }
            
            return processed_tweet
            
        except Exception as e:
            logger.error("Error processing tweet", error=e, tweet_id=getattr(tweet, 'id', 'unknown'))
            return None
    
    async def stream_tweets(
        self,
        symbols: List[str] = None,
        callback=None
    ):
        """
        Потоковое получение твитов в реальном времени
        
        Args:
            symbols: Символы для мониторинга
            callback: Функция для обработки каждого твита
        """
        if not self.client:
            await self.initialize()
        
        if not symbols:
            symbols = list(self.crypto_symbols)[:5]
        
        # Построение фильтров
        track_terms = []
        for symbol in symbols:
            track_terms.extend([f"${symbol}", f"#{symbol}", f"{symbol} crypto"])
        
        logger.info(
            "Starting Twitter stream",
            symbols=symbols,
            track_terms_count=len(track_terms)
        )
        
        try:
            # Это упрощенная версия - для production нужен TwitterStream
            while True:
                # В реальной реализации здесь был бы TwitterStream
                tweets = await self.search_tweets(symbols=symbols, limit=10, hours_back=1)
                
                for tweet in tweets:
                    if callback:
                        try:
                            await callback(tweet)
                        except Exception as e:
                            logger.error("Error in stream callback", error=e)
                
                # Пауза между итерациями
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error("Error in Twitter stream", error=e)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики источника
        
        Returns:
            Dict[str, Any]: Статистика работы
        """
        return {
            "source": "twitter",
            "tweets_processed": self.tweets_processed,
            "api_calls_made": self.api_calls_made,
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "last_error": str(self.last_error) if self.last_error else None,
            "crypto_symbols_tracked": len(self.crypto_symbols),
            "crypto_keywords_tracked": len(self.crypto_keywords),
            "initialized": self.client is not None
        }


async def create_twitter_source() -> TwitterSentimentSource:
    """
    Factory function для создания Twitter source
    
    Returns:
        TwitterSentimentSource: Настроенный источник данных
    """
    source = TwitterSentimentSource()
    await source.initialize()
    return source