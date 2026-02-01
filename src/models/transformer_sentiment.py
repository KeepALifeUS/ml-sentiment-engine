"""
Transformer-based Sentiment Analysis для ML-Framework ML Sentiment Engine

Enterprise-grade BERT/FinBERT sentiment analysis с Context7 patterns и async support.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.validators import SentimentScore, TextContent, validate_text_content, sanitize_text

logger = get_logger(__name__)


class TransformerModelManager:
    """Менеджер для управления transformer моделями"""
    
    def __init__(self):
        """Инициализация менеджера моделей"""
        config = get_config()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.ml.use_gpu else "cpu")
        self.models_cache = {}
        self.tokenizers_cache = {}
        self.pipelines_cache = {}
        
        # Конфигурация моделей
        self.model_configs = {
            "finbert": {
                "model_name": "ProsusAI/finbert",
                "labels": ["negative", "neutral", "positive"],
                "task": "sentiment-analysis"
            },
            "finbert_esg": {
                "model_name": "ProsusAI/finbert-esg",
                "labels": ["negative", "neutral", "positive"],
                "task": "sentiment-analysis"
            },
            "cardiffnlp_twitter": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "labels": ["negative", "neutral", "positive"],
                "task": "sentiment-analysis"
            },
            "distilbert_financetext": {
                "model_name": "ahmedrachid/FinanceInc-Sentiment-Analysis",
                "labels": ["negative", "neutral", "positive"],
                "task": "sentiment-analysis"
            }
        }
        
        # Performance metrics
        self.predictions_made = 0
        self.total_inference_time = 0.0
        self.model_loads = 0
        self.cache_hits = 0
        
        logger.info(f"Transformer models manager initialized on device: {self.device}")
    
    async def load_model(self, model_key: str) -> Tuple[Any, Any]:
        """
        Загрузка модели и tokenizer
        
        Args:
            model_key: Ключ модели из конфигурации
            
        Returns:
            Tuple[Any, Any]: (model, tokenizer)
        """
        if model_key in self.models_cache:
            self.cache_hits += 1
            return self.models_cache[model_key], self.tokenizers_cache[model_key]
        
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model key: {model_key}")
        
        config = self.model_configs[model_key]
        model_name = config["model_name"]
        
        try:
            start_time = time.time()
            
            # Загрузка в отдельном executor для блокирующих операций
            loop = asyncio.get_event_loop()
            
            tokenizer = await loop.run_in_executor(
                None,
                lambda: AutoTokenizer.from_pretrained(model_name)
            )
            
            model = await loop.run_in_executor(
                None,
                lambda: AutoModelForSequenceClassification.from_pretrained(model_name)
            )
            
            # Перенос на device
            model = model.to(self.device)
            model.eval()  # Режим inference
            
            # Кэширование
            self.models_cache[model_key] = model
            self.tokenizers_cache[model_key] = tokenizer
            self.model_loads += 1
            
            load_time = time.time() - start_time
            logger.info(
                f"Model loaded successfully",
                model_key=model_key,
                model_name=model_name,
                device=str(self.device),
                load_time_s=round(load_time, 2)
            )
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_key}", error=e)
            raise
    
    async def load_pipeline(self, model_key: str) -> Any:
        """
        Загрузка pipeline для модели
        
        Args:
            model_key: Ключ модели
            
        Returns:
            Any: HuggingFace pipeline
        """
        if model_key in self.pipelines_cache:
            self.cache_hits += 1
            return self.pipelines_cache[model_key]
        
        if model_key not in self.model_configs:
            raise ValueError(f"Unknown model key: {model_key}")
        
        config = self.model_configs[model_key]
        
        try:
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            
            sentiment_pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    config["task"],
                    model=config["model_name"],
                    device=0 if self.device.type == "cuda" else -1
                )
            )
            
            self.pipelines_cache[model_key] = sentiment_pipeline
            
            load_time = time.time() - start_time
            logger.info(
                f"Pipeline loaded successfully",
                model_key=model_key,
                load_time_s=round(load_time, 2)
            )
            
            return sentiment_pipeline
            
        except Exception as e:
            logger.error(f"Failed to load pipeline for {model_key}", error=e)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики менеджера моделей"""
        return {
            "device": str(self.device),
            "models_loaded": len(self.models_cache),
            "pipelines_loaded": len(self.pipelines_cache),
            "predictions_made": self.predictions_made,
            "avg_inference_time_ms": (
                (self.total_inference_time / max(self.predictions_made, 1)) * 1000
            ),
            "model_loads": self.model_loads,
            "cache_hits": self.cache_hits,
            "available_models": list(self.model_configs.keys())
        }


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer для финансовых текстов
    """
    
    def __init__(self, model_manager: TransformerModelManager):
        """
        Инициализация FinBERT анализатора
        
        Args:
            model_manager: Менеджер моделей
        """
        self.model_manager = model_manager
        self.model_key = "finbert"
        self.model = None
        self.tokenizer = None
        
        # Маппинг лейблов на числовые значения
        self.label_mapping = {
            "negative": -1.0,
            "neutral": 0.0,
            "positive": 1.0
        }
    
    async def initialize(self):
        """Инициализация модели"""
        self.model, self.tokenizer = await self.model_manager.load_model(self.model_key)
        logger.info("FinBERT sentiment analyzer initialized")
    
    async def predict(self, text: str) -> SentimentScore:
        """
        Предсказание sentiment для текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            SentimentScore: Результат анализа
        """
        if not self.model or not self.tokenizer:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Предобработка текста
            cleaned_text = sanitize_text(text)
            if not validate_text_content(cleaned_text, "transformer"):
                raise ValueError("Invalid text content")
            
            # Токенизация
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Перенос на device
            inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)
            
            # Получение вероятностей
            probs = predictions.cpu().numpy()[0]
            
            # Поиск наиболее вероятного класса
            predicted_class_id = np.argmax(probs)
            predicted_label = self.model_manager.model_configs[self.model_key]["labels"][predicted_class_id]
            
            # Конвертация в sentiment score
            sentiment_value = self.label_mapping[predicted_label]
            confidence = float(probs[predicted_class_id])
            
            # Метрики
            inference_time = time.time() - start_time
            self.model_manager.predictions_made += 1
            self.model_manager.total_inference_time += inference_time
            
            logger.debug(
                "FinBERT prediction completed",
                text_length=len(cleaned_text),
                predicted_label=predicted_label,
                confidence=confidence,
                inference_time_ms=round(inference_time * 1000, 2)
            )
            
            return SentimentScore(
                value=sentiment_value,
                confidence=confidence,
                model_name="finbert"
            )
            
        except Exception as e:
            logger.error("FinBERT prediction failed", error=e)
            raise
    
    async def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[SentimentScore]:
        """
        Batch prediction для списка текстов
        
        Args:
            texts: Список текстов для анализа
            batch_size: Размер batch
            
        Returns:
            List[SentimentScore]: Результаты анализа
        """
        if not self.model or not self.tokenizer:
            await self.initialize()
        
        results = []
        
        # Обработка батчами
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                start_time = time.time()
                
                # Предобработка
                cleaned_texts = [sanitize_text(text) for text in batch_texts]
                valid_texts = [text for text in cleaned_texts if text and len(text) > 0]
                
                if not valid_texts:
                    # Добавляем пустые результаты для недействительных текстов
                    results.extend([
                        SentimentScore(value=0.0, confidence=0.0, model_name="finbert")
                        for _ in batch_texts
                    ])
                    continue
                
                # Токенизация batch
                inputs = self.tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Перенос на device
                inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
                
                # Batch inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = F.softmax(outputs.logits, dim=-1)
                
                probs = predictions.cpu().numpy()
                
                # Обработка результатов
                batch_results = []
                for j, prob in enumerate(probs):
                    predicted_class_id = np.argmax(prob)
                    predicted_label = self.model_manager.model_configs[self.model_key]["labels"][predicted_class_id]
                    
                    sentiment_value = self.label_mapping[predicted_label]
                    confidence = float(prob[predicted_class_id])
                    
                    batch_results.append(SentimentScore(
                        value=sentiment_value,
                        confidence=confidence,
                        model_name="finbert"
                    ))
                
                results.extend(batch_results)
                
                # Метрики
                inference_time = time.time() - start_time
                self.model_manager.predictions_made += len(batch_texts)
                self.model_manager.total_inference_time += inference_time
                
                logger.debug(
                    "FinBERT batch prediction completed",
                    batch_size=len(batch_texts),
                    inference_time_ms=round(inference_time * 1000, 2)
                )
                
            except Exception as e:
                logger.error("FinBERT batch prediction failed", error=e)
                # Добавляем нулевые результаты для неудачного batch
                results.extend([
                    SentimentScore(value=0.0, confidence=0.0, model_name="finbert")
                    for _ in batch_texts
                ])
        
        return results


class TwitterRoBERTaSentimentAnalyzer:
    """
    RoBERTa-based sentiment analyzer специально для Twitter/социальных медиа
    """
    
    def __init__(self, model_manager: TransformerModelManager):
        """Инициализация Twitter RoBERTa анализатора"""
        self.model_manager = model_manager
        self.model_key = "cardiffnlp_twitter"
        self.pipeline = None
    
    async def initialize(self):
        """Инициализация pipeline"""
        self.pipeline = await self.model_manager.load_pipeline(self.model_key)
        logger.info("Twitter RoBERTa sentiment analyzer initialized")
    
    async def predict(self, text: str) -> SentimentScore:
        """
        Предсказание sentiment для социальных медиа текста
        
        Args:
            text: Текст для анализа
            
        Returns:
            SentimentScore: Результат анализа
        """
        if not self.pipeline:
            await self.initialize()
        
        try:
            start_time = time.time()
            
            # Предобработка для социальных медиа
            cleaned_text = self._preprocess_social_text(text)
            
            if not validate_text_content(cleaned_text, "twitter"):
                raise ValueError("Invalid social media text content")
            
            # Inference через pipeline
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.pipeline(cleaned_text)
            )
            
            # Обработка результата
            label = result["label"].lower()
            confidence = result["score"]
            
            # Маппинг labels
            label_mapping = {
                "negative": -1.0,
                "neutral": 0.0, 
                "positive": 1.0
            }
            
            sentiment_value = label_mapping.get(label, 0.0)
            
            # Метрики
            inference_time = time.time() - start_time
            self.model_manager.predictions_made += 1
            self.model_manager.total_inference_time += inference_time
            
            logger.debug(
                "Twitter RoBERTa prediction completed",
                predicted_label=label,
                confidence=confidence,
                inference_time_ms=round(inference_time * 1000, 2)
            )
            
            return SentimentScore(
                value=sentiment_value,
                confidence=confidence,
                model_name="twitter_roberta"
            )
            
        except Exception as e:
            logger.error("Twitter RoBERTa prediction failed", error=e)
            raise
    
    def _preprocess_social_text(self, text: str) -> str:
        """
        Предобработка текста для социальных медиа
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Обработанный текст
        """
        # Базовая очистка
        cleaned = sanitize_text(text)
        
        # Социальные медиа специфичная обработка
        # Нормализация URLs
        cleaned = re.sub(r'http[s]?://[^\s]+', '[URL]', cleaned)
        
        # Нормализация mentions
        cleaned = re.sub(r'@\w+', '[USER]', cleaned)
        
        # Сохранение эмоджи (важно для sentiment)
        # Нормализация повторяющихся символов
        cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
        
        return cleaned


class TransformerSentimentEnsemble:
    """
    Ensemble transformer моделей для robust sentiment analysis
    """
    
    def __init__(self):
        """Инициализация ensemble"""
        self.model_manager = TransformerModelManager()
        self.finbert_analyzer = FinBERTSentimentAnalyzer(self.model_manager)
        self.twitter_analyzer = TwitterRoBERTaSentimentAnalyzer(self.model_manager)
        
        # Веса для ensemble (можно настраивать)
        self.weights = {
            "finbert": 0.6,  # Больший вес для финансовых текстов
            "twitter_roberta": 0.4  # Меньший вес для социальных медиа
        }
    
    async def initialize(self):
        """Инициализация всех анализаторов"""
        await self.finbert_analyzer.initialize()
        await self.twitter_analyzer.initialize()
        logger.info("Transformer sentiment ensemble initialized")
    
    async def predict(
        self, 
        text: str, 
        source: str = "unknown",
        use_weighted_ensemble: bool = True
    ) -> SentimentScore:
        """
        Ensemble prediction с адаптивными весами
        
        Args:
            text: Текст для анализа
            source: Источник текста для адаптации весов
            use_weighted_ensemble: Использовать ли взвешенное усреднение
            
        Returns:
            SentimentScore: Результат ensemble анализа
        """
        try:
            start_time = time.time()
            
            # Получение предсказаний от всех моделей
            finbert_result = await self.finbert_analyzer.predict(text)
            twitter_result = await self.twitter_analyzer.predict(text)
            
            if not use_weighted_ensemble:
                # Простое усреднение
                avg_sentiment = (finbert_result.value + twitter_result.value) / 2
                avg_confidence = (finbert_result.confidence + twitter_result.confidence) / 2
            else:
                # Адаптивные веса в зависимости от источника
                weights = self._adapt_weights_for_source(source)
                
                # Взвешенное усреднение
                avg_sentiment = (
                    finbert_result.value * weights["finbert"] +
                    twitter_result.value * weights["twitter_roberta"]
                )
                
                avg_confidence = (
                    finbert_result.confidence * weights["finbert"] +
                    twitter_result.confidence * weights["twitter_roberta"]
                )
            
            inference_time = time.time() - start_time
            
            logger.debug(
                "Transformer ensemble prediction completed",
                source=source,
                finbert_sentiment=finbert_result.value,
                twitter_sentiment=twitter_result.value,
                ensemble_sentiment=avg_sentiment,
                ensemble_confidence=avg_confidence,
                inference_time_ms=round(inference_time * 1000, 2)
            )
            
            return SentimentScore(
                value=avg_sentiment,
                confidence=avg_confidence,
                model_name="transformer_ensemble"
            )
            
        except Exception as e:
            logger.error("Transformer ensemble prediction failed", error=e)
            raise
    
    def _adapt_weights_for_source(self, source: str) -> Dict[str, float]:
        """
        Адаптация весов в зависимости от источника данных
        
        Args:
            source: Источник данных
            
        Returns:
            Dict[str, float]: Адаптированные веса
        """
        if source in ["twitter", "reddit", "telegram", "discord"]:
            # Больший вес социальным медиа моделям
            return {
                "finbert": 0.3,
                "twitter_roberta": 0.7
            }
        elif source in ["news", "bloomberg", "reuters"]:
            # Больший вес финансовым моделям
            return {
                "finbert": 0.8,
                "twitter_roberta": 0.2
            }
        else:
            # Базовые веса
            return self.weights
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики ensemble"""
        base_stats = self.model_manager.get_stats()
        base_stats["ensemble_weights"] = self.weights
        base_stats["model_type"] = "transformer_ensemble"
        return base_stats


# Factory functions
async def create_finbert_analyzer() -> FinBERTSentimentAnalyzer:
    """
    Factory для создания FinBERT анализатора
    
    Returns:
        FinBERTSentimentAnalyzer: Инициализированный анализатор
    """
    manager = TransformerModelManager()
    analyzer = FinBERTSentimentAnalyzer(manager)
    await analyzer.initialize()
    return analyzer


async def create_twitter_analyzer() -> TwitterRoBERTaSentimentAnalyzer:
    """
    Factory для создания Twitter анализатора
    
    Returns:
        TwitterRoBERTaSentimentAnalyzer: Инициализированный анализатор
    """
    manager = TransformerModelManager()
    analyzer = TwitterRoBERTaSentimentAnalyzer(manager)
    await analyzer.initialize()
    return analyzer


async def create_transformer_ensemble() -> TransformerSentimentEnsemble:
    """
    Factory для создания transformer ensemble
    
    Returns:
        TransformerSentimentEnsemble: Инициализированный ensemble
    """
    ensemble = TransformerSentimentEnsemble()
    await ensemble.initialize()
    return ensemble