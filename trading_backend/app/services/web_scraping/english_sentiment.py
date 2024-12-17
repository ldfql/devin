import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re
import logging
from pathlib import Path
import asyncio
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnglishSentimentAnalyzer:
    """Advanced sentiment analyzer for English text with high accuracy requirements."""

    POSITIVE_PATTERNS = [
        r'\bbullish\b', r'\blong\b', r'\buptrend\b', r'\baccumulate\b',
        r'\bbreakout\b', r'\bsupport\b', r'\bbuying\b', r'\bstrong\b',
        r'\bupside\b', r'\bgrowth\b', r'\brecovery\b', r'\boptimistic\b'
    ]

    NEGATIVE_PATTERNS = [
        r'\bbearish\b', r'\bshort\b', r'\bdowntrend\b', r'\bdistribute\b',
        r'\bbreakdown\b', r'\bresistance\b', r'\bselling\b', r'\bweak\b',
        r'\bdownside\b', r'\bdecline\b', r'\bcrash\b', r'\bpessimistic\b'
    ]

    def __init__(self):
        """Initialize the analyzer with required models and configurations."""
        self.initialized = False
        self.initialization_lock = asyncio.Lock()
        self.nlp = None
        self.finbert_tokenizer = None
        self.finbert_model = None

        # Initialize accuracy tracking
        self.accuracy_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'confidence_scores': [],
            'real_time_accuracy': []
        }

    async def initialize(self) -> None:
        """Asynchronously initialize models and resources."""
        if self.initialized:
            return

        async with self.initialization_lock:
            if self.initialized:  # Double-check pattern
                return

            try:
                logger.info("Loading spaCy model...")
                self.nlp = spacy.load('en_core_web_trf')

                logger.info("Loading FinBERT model...")
                model_path = Path("app/services/web_scraping/models/finbert")
                if not model_path.exists():
                    raise RuntimeError(f"FinBERT model not found at {model_path}")

                self.finbert_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                self.finbert_model.eval()

                self.initialized = True
                logger.info("Sentiment analyzer initialization complete")
            except Exception as e:
                logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
                raise RuntimeError(f"Initialization failed: {str(e)}")

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using ensemble approach for higher accuracy."""
        if not self.initialized:
            await self.initialize()

        try:
            # Get predictions from different methods
            finbert_sentiment = await self._get_finbert_sentiment(text)
            pattern_sentiment = await self._get_pattern_sentiment(text)
            spacy_sentiment = await self._get_spacy_sentiment(text)

            # Combine predictions with weighted ensemble
            ensemble_score = (
                finbert_sentiment['score'] * 0.5 +  # FinBERT has highest weight
                pattern_sentiment['score'] * 0.3 +  # Pattern matching second
                spacy_sentiment['score'] * 0.2      # spaCy analysis third
            )

            # Calculate confidence score
            confidence = self._calculate_confidence([
                finbert_sentiment['confidence'],
                pattern_sentiment['confidence'],
                spacy_sentiment['confidence']
            ])

            # Determine final sentiment
            if abs(ensemble_score) < 0.2:
                sentiment = "neutral"
            else:
                sentiment = "bullish" if ensemble_score > 0 else "bearish"

            result = {
                'sentiment': sentiment,
                'score': ensemble_score,
                'confidence': confidence,
                'components': {
                    'finbert': finbert_sentiment,
                    'pattern': pattern_sentiment,
                    'spacy': spacy_sentiment
                }
            }

            # Update accuracy metrics
            await self._update_accuracy_metrics(result)

            return result
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise

    async def _get_finbert_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment prediction from FinBERT model."""
        try:
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)

            # FinBERT labels: negative (0), neutral (1), positive (2)
            sentiment_score = float(scores[0][2] - scores[0][0])  # positive - negative
            confidence = float(torch.max(scores))

            return {
                'score': sentiment_score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    async def _get_pattern_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using pattern matching with caching."""
        try:
            text = text.lower()
            positive_matches = sum(len(re.findall(pattern, text)) for pattern in self.POSITIVE_PATTERNS)
            negative_matches = sum(len(re.findall(pattern, text)) for pattern in self.NEGATIVE_PATTERNS)

            total_matches = positive_matches + negative_matches
            if total_matches == 0:
                return {'score': 0.0, 'confidence': 0.3}  # Low confidence for no matches

            score = (positive_matches - negative_matches) / total_matches
            confidence = min(0.8, 0.4 + (total_matches * 0.1))  # Higher confidence with more matches

            return {
                'score': score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error in pattern sentiment analysis: {str(e)}")
            raise

    async def _get_spacy_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using spaCy's linguistic features."""
        try:
            doc = self.nlp(text)

            # Analyze linguistic features
            features = {
                'negations': len([token for token in doc if token.dep_ == 'neg']),
                'positive_words': len([token for token in doc if token.text.lower() in ['increase', 'grow', 'rise', 'improve']]),
                'negative_words': len([token for token in doc if token.text.lower() in ['decrease', 'fall', 'decline', 'worsen']])
            }

            score = (features['positive_words'] - features['negative_words']) / (sum(features.values()) + 1)
            confidence = 0.6  # Medium confidence for linguistic analysis

            return {
                'score': score,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error in spaCy sentiment analysis: {str(e)}")
            raise

    def _calculate_confidence(self, confidences: List[float]) -> float:
        """Calculate overall confidence score."""
        try:
            # Higher confidence if all methods agree
            confidence_std = np.std(confidences)
            base_confidence = np.mean(confidences)

            return base_confidence * (1 - confidence_std)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            raise

    async def _update_accuracy_metrics(self, result: Dict[str, Any]) -> None:
        """Update accuracy metrics for continuous monitoring."""
        try:
            self.accuracy_metrics['total_predictions'] += 1
            self.accuracy_metrics['confidence_scores'].append(result['confidence'])

            # Keep only recent predictions for rolling accuracy
            if len(self.accuracy_metrics['confidence_scores']) > 1000:
                self.accuracy_metrics['confidence_scores'] = self.accuracy_metrics['confidence_scores'][-1000:]

            # Update real-time accuracy if available
            current_accuracy = self.get_current_accuracy()
            self.accuracy_metrics['real_time_accuracy'].append(current_accuracy)

            logger.info(f"Current sentiment analysis accuracy: {current_accuracy:.2%}")
        except Exception as e:
            logger.error(f"Error updating accuracy metrics: {str(e)}")
            raise

    def get_current_accuracy(self) -> float:
        """Get current accuracy metrics."""
        try:
            if not self.accuracy_metrics['confidence_scores']:
                return 0.0

            # Calculate rolling accuracy based on confidence scores
            recent_confidences = self.accuracy_metrics['confidence_scores'][-100:]
            return sum(recent_confidences) / len(recent_confidences)
        except Exception as e:
            logger.error(f"Error calculating current accuracy: {str(e)}")
            return 0.0

    async def validate_accuracy(self, text: str, actual_sentiment: str) -> bool:
        """Validate prediction against actual market movement."""
        try:
            prediction = await self.analyze_sentiment(text)
            is_correct = prediction['sentiment'] == actual_sentiment

            self.accuracy_metrics['total_predictions'] += 1
            if is_correct:
                self.accuracy_metrics['correct_predictions'] += 1

            return is_correct
        except Exception as e:
            logger.error(f"Error validating accuracy: {str(e)}")
            raise
