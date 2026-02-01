"""
ML-Framework ML Sentiment Engine - Comprehensive Sentiment Aggregation for Crypto Trading

Enterprise-grade sentiment analysis system with Context7 patterns for real-time
crypto market sentiment aggregation from multiple sources.
"""

from .src.api.sentiment_api import SentimentAPI
from .src.aggregation.ensemble_aggregator import EnsembleAggregator
from .src.models.ensemble_sentiment import EnsembleSentimentModel
from .src.indicators.fear_greed_index import CryptoFearGreedIndex
from .src.storage.sentiment_database import SentimentDatabase

__version__ = "1.0.0"
__author__ = "ML-Framework ML Team"

__all__ = [
    "SentimentAPI",
    "EnsembleAggregator", 
    "EnsembleSentimentModel",
    "CryptoFearGreedIndex",
    "SentimentDatabase",
]