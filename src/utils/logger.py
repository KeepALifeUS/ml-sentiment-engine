"""
Structured logging для ML-Framework ML Sentiment Engine

Enterprise-grade logging с OpenTelemetry integration и Context7 patterns.
"""

import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import get_config, LogLevel

# Context variables для корреляции запросов
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
symbol_ctx: ContextVar[Optional[str]] = ContextVar("symbol", default=None)


def add_correlation_context(
    logger: structlog.BoundLogger, 
    name: str, 
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Добавление correlation контекста в лог записи
    
    Args:
        logger: Структурированный логгер
        name: Имя logger
        event_dict: Словарь события
        
    Returns:
        Dict[str, Any]: Обогащенный словарь события
    """
    event_dict["request_id"] = request_id_ctx.get()
    event_dict["user_id"] = user_id_ctx.get()
    event_dict["symbol"] = symbol_ctx.get()
    
    # Добавление trace информации
    span = trace.get_current_span()
    if span:
        span_ctx = span.get_span_context()
        event_dict["trace_id"] = format(span_ctx.trace_id, "032x")
        event_dict["span_id"] = format(span_ctx.span_id, "016x")
    
    return event_dict


def add_service_context(
    logger: structlog.BoundLogger, 
    name: str, 
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Добавление service контекста
    
    Args:
        logger: Структурированный логгер
        name: Имя logger
        event_dict: Словарь события
        
    Returns:
        Dict[str, Any]: Обогащенный словарь события
    """
    config = get_config()
    event_dict["service"] = config.service_name
    event_dict["version"] = config.service_version
    event_dict["environment"] = config.environment.value
    
    return event_dict


def add_timestamp(
    logger: structlog.BoundLogger, 
    name: str, 
    event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Добавление timestamp в ISO формате
    
    Args:
        logger: Структурированный логгер
        name: Имя logger
        event_dict: Словарь события
        
    Returns:
        Dict[str, Any]: Обогащенный словарь события
    """
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict


def setup_tracing():
    """Настройка OpenTelemetry tracing"""
    config = get_config()
    
    if not config.monitoring.tracing_enabled or not config.monitoring.jaeger_endpoint:
        return
    
    # Создание resource
    resource = Resource(attributes={
        SERVICE_NAME: config.service_name
    })
    
    # Настройка tracer provider
    provider = TracerProvider(resource=resource)
    
    # Добавление Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=config.monitoring.jaeger_endpoint.split("://")[1].split(":")[0],
        agent_port=int(config.monitoring.jaeger_endpoint.split(":")[-1]),
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(span_processor)
    
    # Установка global tracer provider
    trace.set_tracer_provider(provider)
    
    # Включение автоматической инструментации логирования
    LoggingInstrumentor().instrument(set_logging_format=True)


def setup_logging():
    """
    Настройка структурированного логирования
    
    Инициализация structlog с enterprise patterns:
    - JSON formatting для production
    - Correlation IDs
    - OpenTelemetry integration
    - Performance monitoring
    """
    config = get_config()
    
    # Настройка уровня логирования
    log_level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_map[config.log_level])
    
    # Удаление существующих handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Настройка processors
    processors = [
        structlog.stdlib.filter_by_level,
        add_service_context,
        add_correlation_context,
        add_timestamp,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]
    
    # Добавление stacktrace для ошибок
    processors.append(structlog.processors.StackInfoRenderer())
    processors.append(structlog.dev.set_exc_info)
    
    # Выбор renderer в зависимости от среды
    if config.is_development:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # Конфигурация structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Настройка console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level_map[config.log_level])
    
    # Формат для обычных логов
    if config.is_development:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        formatter = logging.Formatter("%(message)s")
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Настройка tracing
    setup_tracing()


class SentimentLogger:
    """
    Специализированный logger для sentiment analysis
    
    Provides:
    - Typed logging methods
    - Performance monitoring
    - Error tracking
    - Business metrics logging
    """
    
    def __init__(self, name: str):
        """
        Инициализация logger
        
        Args:
            name: Имя logger (обычно __name__)
        """
        self.logger = structlog.get_logger(name)
    
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info level logging"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Error level logging с exception tracking"""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
            self.logger.error(message, exc_info=error, **kwargs)
        else:
            self.logger.error(message, **kwargs)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Critical level logging"""
        if error:
            kwargs["error_type"] = type(error).__name__
            kwargs["error_message"] = str(error)
            self.logger.critical(message, exc_info=error, **kwargs)
        else:
            self.logger.critical(message, **kwargs)
    
    def sentiment_processed(
        self, 
        symbol: str, 
        source: str, 
        sentiment_score: float, 
        processing_time_ms: float,
        **kwargs
    ):
        """Логирование обработанного sentiment"""
        self.logger.info(
            "Sentiment processed",
            event_type="sentiment_processed",
            symbol=symbol,
            source=source,
            sentiment_score=sentiment_score,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
    
    def model_prediction(
        self,
        model_name: str,
        input_text: str,
        prediction: float,
        confidence: float,
        processing_time_ms: float,
        **kwargs
    ):
        """Логирование ML model prediction"""
        self.logger.info(
            "Model prediction completed",
            event_type="model_prediction",
            model_name=model_name,
            input_length=len(input_text),
            prediction=prediction,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
    
    def api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        processing_time_ms: float,
        **kwargs
    ):
        """Логирование API request"""
        self.logger.info(
            "API request processed",
            event_type="api_request",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
    
    def data_source_fetch(
        self,
        source: str,
        symbol: str,
        records_fetched: int,
        processing_time_ms: float,
        **kwargs
    ):
        """Логирование получения данных из источника"""
        self.logger.info(
            "Data source fetch completed",
            event_type="data_source_fetch",
            source=source,
            symbol=symbol,
            records_fetched=records_fetched,
            processing_time_ms=processing_time_ms,
            **kwargs
        )
    
    def circuit_breaker_opened(
        self,
        service: str,
        failure_count: int,
        **kwargs
    ):
        """Логирование открытия circuit breaker"""
        self.logger.warning(
            "Circuit breaker opened",
            event_type="circuit_breaker_opened",
            service=service,
            failure_count=failure_count,
            **kwargs
        )
    
    def performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        **kwargs
    ):
        """Логирование performance метрики"""
        self.logger.info(
            "Performance metric recorded",
            event_type="performance_metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            **kwargs
        )


def get_logger(name: str) -> SentimentLogger:
    """
    Создание специализированного logger
    
    Args:
        name: Имя logger (обычно __name__)
        
    Returns:
        SentimentLogger: Настроенный logger
    """
    return SentimentLogger(name)


def set_request_context(request_id: str, user_id: Optional[str] = None, symbol: Optional[str] = None):
    """
    Установка контекста запроса
    
    Args:
        request_id: Уникальный ID запроса
        user_id: ID пользователя (опционально)
        symbol: Криптовалютный символ (опционально)
    """
    request_id_ctx.set(request_id)
    if user_id:
        user_id_ctx.set(user_id)
    if symbol:
        symbol_ctx.set(symbol)


def clear_request_context():
    """Очистка контекста запроса"""
    request_id_ctx.set(None)
    user_id_ctx.set(None)
    symbol_ctx.set(None)


# Инициализация логирования при импорте модуля
setup_logging()