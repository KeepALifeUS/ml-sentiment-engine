"""
Data validation utilities for ML-Framework ML Sentiment Engine

Enterprise-grade валидация с Context7 patterns и типизированными проверками.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from pydantic.validators import float_validator, int_validator


class SentimentScore(BaseModel):
    """Валидированная sentiment оценка"""
    
    value: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score от -1 до 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score от 0 до 1")
    model_name: str = Field(..., min_length=1, max_length=100, description="Название модели")
    
    @validator("value")
    def validate_sentiment_value(cls, v):
        """Валидация sentiment значения"""
        if not isinstance(v, (int, float)):
            raise ValueError("Sentiment value must be numeric")
        if not -1.0 <= v <= 1.0:
            raise ValueError("Sentiment value must be between -1.0 and 1.0")
        return float(v)
    
    @validator("confidence")
    def validate_confidence(cls, v):
        """Валидация confidence значения"""
        if not isinstance(v, (int, float)):
            raise ValueError("Confidence must be numeric")
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return float(v)


class CryptoSymbol(BaseModel):
    """Валидированный криптовалютный символ"""
    
    symbol: str = Field(..., min_length=2, max_length=10, description="Символ криптовалюты")
    base: Optional[str] = Field(None, min_length=2, max_length=10, description="Базовая валюта")
    quote: Optional[str] = Field(None, min_length=2, max_length=10, description="Котируемая валюта")
    
    @validator("symbol", pre=True)
    def validate_symbol(cls, v):
        """Валидация символа криптовалюты"""
        if not isinstance(v, str):
            raise ValueError("Symbol must be a string")
        
        # Приведение к верхнему регистру
        symbol = v.upper().strip()
        
        # Проверка формата
        if not re.match(r"^[A-Z0-9]{2,10}$", symbol):
            raise ValueError("Invalid crypto symbol format")
        
        return symbol
    
    @validator("base", "quote", pre=True)
    def validate_currency(cls, v):
        """Валидация валютных кодов"""
        if v is None:
            return v
        
        if not isinstance(v, str):
            raise ValueError("Currency code must be a string")
        
        currency = v.upper().strip()
        if not re.match(r"^[A-Z0-9]{2,10}$", currency):
            raise ValueError("Invalid currency code format")
        
        return currency


class TextContent(BaseModel):
    """Валидированный текстовый контент"""
    
    text: str = Field(..., min_length=1, max_length=10000, description="Текстовый контент")
    language: Optional[str] = Field(None, max_length=5, description="Код языка")
    source: str = Field(..., min_length=1, max_length=50, description="Источник контента")
    created_at: Optional[datetime] = Field(None, description="Время создания")
    
    @validator("text", pre=True)
    def validate_text(cls, v):
        """Валидация текста"""
        if not isinstance(v, str):
            raise ValueError("Text must be a string")
        
        text = v.strip()
        if not text:
            raise ValueError("Text cannot be empty")
        
        if len(text) > 10000:
            raise ValueError("Text too long (max 10000 characters)")
        
        return text
    
    @validator("language", pre=True)
    def validate_language(cls, v):
        """Валидация языкового кода"""
        if v is None:
            return v
        
        if not isinstance(v, str):
            raise ValueError("Language code must be a string")
        
        lang = v.lower().strip()
        if not re.match(r"^[a-z]{2}(-[a-z]{2})?$", lang):
            raise ValueError("Invalid language code format (expected: 'en' or 'en-us')")
        
        return lang
    
    @validator("source", pre=True)
    def validate_source(cls, v):
        """Валидация источника"""
        if not isinstance(v, str):
            raise ValueError("Source must be a string")
        
        source = v.strip().lower()
        allowed_sources = [
            "twitter", "reddit", "telegram", "discord", "news", 
            "web", "blog", "forum", "chat", "social"
        ]
        
        if source not in allowed_sources:
            raise ValueError(f"Invalid source. Allowed: {', '.join(allowed_sources)}")
        
        return source


class TimeRange(BaseModel):
    """Валидированный временной диапазон"""
    
    start_time: datetime = Field(..., description="Начальное время")
    end_time: datetime = Field(..., description="Конечное время")
    
    @validator("end_time")
    def validate_time_range(cls, v, values):
        """Валидация временного диапазона"""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("End time must be after start time")
        
        # Проверка на разумный диапазон (не более 1 года)
        if "start_time" in values:
            diff = v - values["start_time"]
            if diff.days > 365:
                raise ValueError("Time range cannot exceed 1 year")
        
        return v


class APIRequestData(BaseModel):
    """Валидированные данные API запроса"""
    
    method: str = Field(..., description="HTTP метод")
    endpoint: str = Field(..., description="API endpoint")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Параметры запроса")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP заголовки")
    
    @validator("method", pre=True)
    def validate_method(cls, v):
        """Валидация HTTP метода"""
        if not isinstance(v, str):
            raise ValueError("Method must be a string")
        
        method = v.upper().strip()
        allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
        
        if method not in allowed_methods:
            raise ValueError(f"Invalid HTTP method. Allowed: {', '.join(allowed_methods)}")
        
        return method
    
    @validator("endpoint", pre=True)
    def validate_endpoint(cls, v):
        """Валидация endpoint"""
        if not isinstance(v, str):
            raise ValueError("Endpoint must be a string")
        
        endpoint = v.strip()
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        
        # Базовая валидация URL пути
        if not re.match(r"^/[a-zA-Z0-9/_\-\.]*$", endpoint):
            raise ValueError("Invalid endpoint format")
        
        return endpoint


class ModelConfiguration(BaseModel):
    """Валидированная конфигурация ML модели"""
    
    model_name: str = Field(..., min_length=1, max_length=100, description="Название модели")
    model_type: str = Field(..., description="Тип модели")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Параметры модели")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Порог принятия решения")
    
    @validator("model_type", pre=True)
    def validate_model_type(cls, v):
        """Валидация типа модели"""
        if not isinstance(v, str):
            raise ValueError("Model type must be a string")
        
        model_type = v.lower().strip()
        allowed_types = [
            "bert", "finbert", "vader", "textblob", "ensemble", 
            "lstm", "cnn", "transformer", "custom"
        ]
        
        if model_type not in allowed_types:
            raise ValueError(f"Invalid model type. Allowed: {', '.join(allowed_types)}")
        
        return model_type


def validate_crypto_symbol(symbol: str) -> bool:
    """
    Валидация символа криптовалюты
    
    Args:
        symbol: Символ для валидации
        
    Returns:
        bool: True если символ валиден
    """
    try:
        CryptoSymbol(symbol=symbol)
        return True
    except Exception:
        return False


def validate_sentiment_score(score: float, confidence: float = 1.0) -> bool:
    """
    Валидация sentiment оценки
    
    Args:
        score: Sentiment score
        confidence: Confidence score
        
    Returns:
        bool: True если оценка валидна
    """
    try:
        SentimentScore(value=score, confidence=confidence, model_name="validator")
        return True
    except Exception:
        return False


def validate_text_content(text: str, source: str) -> bool:
    """
    Валидация текстового контента
    
    Args:
        text: Текст для валидации
        source: Источник текста
        
    Returns:
        bool: True если контент валиден
    """
    try:
        TextContent(text=text, source=source)
        return True
    except Exception:
        return False


def sanitize_text(text: str) -> str:
    """
    Очистка текста от потенциально опасного контента
    
    Args:
        text: Исходный текст
        
    Returns:
        str: Очищенный текст
    """
    if not isinstance(text, str):
        return ""
    
    # Удаление HTML тегов
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Удаление URL
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)
    
    # Удаление email адресов
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", " ", text)
    
    # Удаление специальных символов (кроме базовых)
    text = re.sub(r"[^\w\s\.\,\!\?\:\;\-\(\)\'\"]+", " ", text)
    
    # Нормализация пробелов
    text = re.sub(r"\s+", " ", text.strip())
    
    return text


def validate_aggregation_weights(weights: Dict[str, float]) -> bool:
    """
    Валидация весов для агрегации
    
    Args:
        weights: Словарь весов
        
    Returns:
        bool: True если веса валидны
    """
    if not isinstance(weights, dict):
        return False
    
    if not weights:
        return False
    
    # Проверка что все веса числовые и положительные
    for weight in weights.values():
        if not isinstance(weight, (int, float)):
            return False
        if weight < 0:
            return False
    
    # Проверка что сумма весов примерно равна 1.0
    total_weight = sum(weights.values())
    if not 0.95 <= total_weight <= 1.05:
        return False
    
    return True


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Нормализация весов к сумме 1.0
    
    Args:
        weights: Исходные веса
        
    Returns:
        Dict[str, float]: Нормализованные веса
    """
    if not weights:
        return {}
    
    total = sum(weights.values())
    if total == 0:
        # Равные веса если сумма 0
        equal_weight = 1.0 / len(weights)
        return {key: equal_weight for key in weights.keys()}
    
    return {key: value / total for key, value in weights.items()}


class ValidationError(Exception):
    """Кастомное исключение для ошибок валидации"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value