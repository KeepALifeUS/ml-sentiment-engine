"""
Telegram Sentiment Data Source для ML-Framework ML Sentiment Engine

Enterprise-grade Telegram data collection с Context7 patterns и async support.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import re

from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from telethon.tl.types import Channel, Chat, User

from ..utils.logger import get_logger
from ..utils.config import get_config, get_crypto_symbols, get_crypto_keywords
from ..utils.validators import TextContent, CryptoSymbol, validate_text_content, sanitize_text

logger = get_logger(__name__)


class TelegramRateLimiter:
    """Rate limiter для Telegram API"""
    
    def __init__(self, messages_per_second: float = 1.0):
        self.messages_per_second = messages_per_second
        self.min_interval = 1.0 / messages_per_second
        self.last_request_time = 0
    
    async def acquire(self):
        """Получение разрешения на запрос"""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()


class TelegramSentimentSource:
    """
    Enterprise-grade Telegram data source для sentiment analysis
    
    Features:
    - Multi-channel monitoring
    - Real-time message streaming
    - Crypto-focused channels
    - Rate limiting compliance
    - Message deduplication
    - Channel metadata tracking
    """
    
    def __init__(self):
        """Инициализация Telegram source"""
        config = get_config()
        
        # Telegram API credentials
        self.api_id = config.social.telegram_api_id
        self.api_hash = config.social.telegram_api_hash
        self.phone = config.social.telegram_phone
        
        # Telegram client
        self.client: Optional[TelegramClient] = None
        
        # Rate limiting
        self.rate_limiter = TelegramRateLimiter()
        
        # Crypto symbols and keywords
        self.crypto_symbols = set(get_crypto_symbols())
        self.crypto_keywords = set(get_crypto_keywords())
        
        # Crypto Telegram channels/groups
        self.crypto_channels = [
            # Public crypto channels
            "@bitcoin",
            "@ethereum", 
            "@binance",
            "@CoinDesk",
            "@cointelegraph",
            "@cryptonews",
            
            # Trading channels
            "@cryptosignals",
            "@binancesignals",
            "@freecryptosignals",
            "@cryptowhales",
            "@whalewatching",
            
            # Analysis channels
            "@cryptoanalysis",
            "@bitcoinanalysis", 
            "@technicalanalysis",
            "@cryptoTA",
            
            # News aggregators
            "@cryptonewsaggregator",
            "@dailycryptonews",
            "@cryptoupdates"
        ]
        
        # Performance metrics
        self.messages_processed = 0
        self.channels_monitored = 0
        self.api_calls_made = 0
        self.last_error = None
        
        # Message deduplication
        self.seen_messages = set()
    
    async def initialize(self):
        """Инициализация Telegram client"""
        if not all([self.api_id, self.api_hash]):
            raise ValueError("Telegram API ID and Hash are required")
        
        self.client = TelegramClient(
            'ml-framework_sentiment_session',
            self.api_id,
            self.api_hash
        )
        
        try:
            await self.client.start(phone=self.phone)
            logger.info("Telegram client initialized successfully")
            
            # Проверка авторизации
            me = await self.client.get_me()
            logger.info(f"Telegram authenticated as: {me.username or me.phone}")
            
        except SessionPasswordNeededError:
            logger.error("Two-factor authentication required for Telegram")
            raise
        except Exception as e:
            logger.error("Failed to initialize Telegram client", error=e)
            raise
    
    async def cleanup(self):
        """Очистка ресурсов"""
        if self.client:
            await self.client.disconnect()
        logger.info("Telegram source cleaned up")
    
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
                rf'\b{symbol}\b',  # Точное совпадение
                rf'\${symbol}\b',  # С префиксом $
                rf'#{symbol}\b',   # Hashtag
                rf'{symbol}/USDT\b',  # Торговые пары
                rf'{symbol}USDT\b',
                rf'{symbol}/BTC\b',
                rf'{symbol}BTC\b'
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    mentioned_symbols.add(symbol)
                    break
        
        return mentioned_symbols
    
    def _is_crypto_relevant(self, text: str) -> bool:
        """
        Проверка релевантности сообщения для крипто-анализа
        
        Args:
            text: Текст для проверки
            
        Returns:
            bool: True если сообщение релевантно
        """
        text_lower = text.lower()
        
        # Проверка ключевых слов
        for keyword in self.crypto_keywords:
            if keyword in text_lower:
                return True
        
        # Проверка символов криптовалют
        if self._extract_crypto_mentions(text):
            return True
        
        # Telegram-специфичные паттерны
        telegram_patterns = [
            r'🚀', r'📈', r'📉', r'💎', r'🌙',  # Криптоэмодзи
            r'\bto\s+the\s+moon\b', r'\bhodl\b', r'\bdip\b',
            r'\bpumping\b', r'\bdumping\b', r'\bwhales?\b',
            r'\bsignal\b', r'\bbuy\b', r'\bsell\b', r'\btarget\b'
        ]
        
        for pattern in telegram_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    async def _get_channel_info(self, channel_username: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о канале
        
        Args:
            channel_username: Username канала
            
        Returns:
            Optional[Dict[str, Any]]: Информация о канале
        """
        try:
            await self.rate_limiter.acquire()
            
            entity = await self.client.get_entity(channel_username)
            
            channel_info = {
                "id": entity.id,
                "username": getattr(entity, 'username', None),
                "title": getattr(entity, 'title', ''),
                "participants_count": getattr(entity, 'participants_count', 0),
                "description": getattr(entity, 'about', ''),
                "type": "channel" if isinstance(entity, Channel) else "chat"
            }
            
            return channel_info
            
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_username}", error=e)
            return None
    
    async def fetch_channel_messages(
        self,
        channel_username: str,
        limit: int = 100,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Получение сообщений из канала
        
        Args:
            channel_username: Username канала
            limit: Максимальное количество сообщений
            hours_back: Период в часах
            
        Returns:
            List[Dict[str, Any]]: Список обработанных сообщений
        """
        if not self.client:
            await self.initialize()
        
        try:
            messages = []
            offset_date = datetime.utcnow() - timedelta(hours=hours_back)
            
            await self.rate_limiter.acquire()
            
            # Получение сообщений
            async for message in self.client.iter_messages(
                channel_username,
                limit=limit,
                offset_date=offset_date
            ):
                processed_message = await self._process_message(message, channel_username)
                if processed_message and self._is_crypto_relevant(processed_message["text"]):
                    messages.append(processed_message)
                
                # Проверка на дубликаты
                message_id = f"{channel_username}_{message.id}"
                if message_id in self.seen_messages:
                    continue
                
                self.seen_messages.add(message_id)
            
            self.messages_processed += len(messages)
            self.api_calls_made += 1
            
            logger.info(
                "Messages fetched from Telegram channel",
                channel=channel_username,
                messages_count=len(messages),
                total_processed=self.messages_processed
            )
            
            return messages
            
        except FloodWaitError as e:
            logger.warning(f"Telegram flood wait for {e.seconds} seconds", channel=channel_username)
            await asyncio.sleep(e.seconds)
            return []
        except Exception as e:
            self.last_error = e
            logger.error(f"Error fetching messages from {channel_username}", error=e)
            return []
    
    async def _process_message(self, message, channel_username: str) -> Optional[Dict[str, Any]]:
        """
        Обработка одного сообщения
        
        Args:
            message: Объект сообщения от Telegram API
            channel_username: Username канала
            
        Returns:
            Optional[Dict[str, Any]]: Обработанное сообщение или None
        """
        try:
            # Проверка наличия текста
            if not message.text:
                return None
            
            text = message.text
            if len(text) < 5:
                return None
            
            # Очистка текста
            cleaned_text = sanitize_text(text)
            if not cleaned_text:
                return None
            
            # Валидация контента
            if not validate_text_content(cleaned_text, "telegram"):
                return None
            
            # Извлечение информации об авторе
            sender = message.sender
            author_info = {
                "id": sender.id if sender else None,
                "username": getattr(sender, 'username', None),
                "first_name": getattr(sender, 'first_name', ''),
                "is_bot": getattr(sender, 'bot', False)
            }
            
            # Метрики взаимодействия
            views = getattr(message, 'views', 0)
            forwards = getattr(message, 'forwards', 0)
            replies = getattr(message, 'replies', None)
            reply_count = replies.replies if replies else 0
            
            processed_message = {
                "id": f"{channel_username}_{message.id}",
                "text": cleaned_text,
                "original_text": text,
                "symbols_mentioned": list(self._extract_crypto_mentions(text)),
                "source": "telegram",
                "channel": channel_username,
                "message_id": message.id,
                "created_at": message.date.isoformat() if message.date else datetime.utcnow().isoformat(),
                "author": author_info,
                "metrics": {
                    "views": views,
                    "forwards": forwards,
                    "replies": reply_count,
                    "engagement_score": views * 0.1 + forwards * 2 + reply_count * 1.5
                },
                "metadata": {
                    "language": "en",  # В основном английский контент
                    "platform": "telegram",
                    "content_type": "message",
                    "has_media": bool(message.media),
                    "is_reply": bool(message.reply_to),
                    "is_forwarded": bool(message.forward)
                }
            }
            
            return processed_message
            
        except Exception as e:
            logger.error("Error processing Telegram message", error=e, message_id=getattr(message, 'id', 'unknown'))
            return None
    
    async def fetch_all_channels(
        self,
        limit_per_channel: int = 50,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Получение сообщений из всех отслеживаемых каналов
        
        Args:
            limit_per_channel: Лимит сообщений на канал
            hours_back: Период в часах
            
        Returns:
            List[Dict[str, Any]]: Список всех сообщений
        """
        all_messages = []
        
        for channel in self.crypto_channels:
            try:
                messages = await self.fetch_channel_messages(
                    channel,
                    limit=limit_per_channel,
                    hours_back=hours_back
                )
                all_messages.extend(messages)
                self.channels_monitored += 1
                
                # Пауза между каналами для соблюдения rate limit
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing Telegram channel {channel}", error=e)
                continue
        
        logger.info(
            "All Telegram channels processed",
            channels_processed=self.channels_monitored,
            total_messages=len(all_messages),
            total_processed=self.messages_processed
        )
        
        return all_messages
    
    async def start_real_time_monitoring(
        self,
        channels: List[str] = None,
        callback=None
    ):
        """
        Запуск мониторинга в реальном времени
        
        Args:
            channels: Список каналов для мониторинга
            callback: Функция для обработки новых сообщений
        """
        if not self.client:
            await self.initialize()
        
        if not channels:
            channels = self.crypto_channels
        
        # Получение entity для каналов
        channel_entities = []
        for channel in channels:
            try:
                entity = await self.client.get_entity(channel)
                channel_entities.append(entity)
            except Exception as e:
                logger.error(f"Error getting entity for {channel}", error=e)
                continue
        
        logger.info(
            "Starting real-time Telegram monitoring",
            channels_count=len(channel_entities)
        )
        
        # Event handler для новых сообщений
        @self.client.on(events.NewMessage(chats=channel_entities))
        async def handle_new_message(event):
            try:
                channel_username = getattr(event.chat, 'username', f'id_{event.chat_id}')
                processed_message = await self._process_message(event.message, channel_username)
                
                if processed_message and self._is_crypto_relevant(processed_message["text"]):
                    self.messages_processed += 1
                    
                    if callback:
                        try:
                            await callback(processed_message)
                        except Exception as e:
                            logger.error("Error in Telegram message callback", error=e)
                    
                    logger.debug(
                        "New crypto message received",
                        channel=channel_username,
                        message_id=processed_message["id"]
                    )
            
            except Exception as e:
                logger.error("Error handling new Telegram message", error=e)
        
        # Запуск клиента (блокирующий вызов)
        try:
            await self.client.run_until_disconnected()
        except Exception as e:
            logger.error("Error in Telegram real-time monitoring", error=e)
            raise
    
    async def search_messages(
        self,
        query: str,
        channels: List[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Поиск сообщений по ключевому слову
        
        Args:
            query: Поисковый запрос
            channels: Каналы для поиска
            limit: Максимальное количество результатов
            
        Returns:
            List[Dict[str, Any]]: Найденные сообщения
        """
        if not channels:
            channels = self.crypto_channels[:5]  # Ограничиваем для поиска
        
        all_results = []
        
        for channel in channels:
            try:
                await self.rate_limiter.acquire()
                
                results = []
                
                # Поиск в сообщениях канала
                async for message in self.client.iter_messages(
                    channel,
                    search=query,
                    limit=limit // len(channels)
                ):
                    processed_message = await self._process_message(message, channel)
                    if processed_message:
                        results.append(processed_message)
                
                all_results.extend(results)
                
                logger.debug(
                    "Telegram search completed for channel",
                    channel=channel,
                    query=query,
                    results_count=len(results)
                )
                
            except Exception as e:
                logger.error(f"Error searching in Telegram channel {channel}", error=e)
                continue
        
        # Сортировка по времени
        all_results.sort(
            key=lambda m: m.get('created_at', ''),
            reverse=True
        )
        
        logger.info(
            "Telegram search completed",
            query=query,
            channels_searched=len(channels),
            total_results=len(all_results)
        )
        
        return all_results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики источника
        
        Returns:
            Dict[str, Any]: Статистика работы
        """
        return {
            "source": "telegram",
            "messages_processed": self.messages_processed,
            "channels_monitored": self.channels_monitored,
            "api_calls_made": self.api_calls_made,
            "last_error": str(self.last_error) if self.last_error else None,
            "crypto_channels_tracked": len(self.crypto_channels),
            "crypto_symbols_tracked": len(self.crypto_symbols),
            "crypto_keywords_tracked": len(self.crypto_keywords),
            "initialized": self.client is not None,
            "seen_messages_count": len(self.seen_messages),
            "channels": self.crypto_channels
        }


async def create_telegram_source() -> TelegramSentimentSource:
    """
    Factory function для создания Telegram source
    
    Returns:
        TelegramSentimentSource: Настроенный источник данных
    """
    source = TelegramSentimentSource()
    await source.initialize()
    return source