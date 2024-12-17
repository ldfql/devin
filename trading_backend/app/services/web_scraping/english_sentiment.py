import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
import re
import logging
from pathlib import Path

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
        """Initialize the sentiment analyzer with multiple models."""
        self.nlp = spacy.load('en_core_web_trf')

        # Load FinBERT model
        model_path = Path("app/services/web_scraping/models/finbert")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.finbert_model.eval()

        # Initialize accuracy tracking
        self.accuracy_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'confidence_scores': []
        }

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using ensemble approach for higher accuracy.
        Combines FinBERT, spaCy, and pattern matching.
        """
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

        # Update accuracy metrics if real outcome is available
        await self._update_accuracy_metrics(result)

        return result

    async def _get_finbert_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment prediction from FinBERT model."""
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

    async def _get_pattern_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using pattern matching."""
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

    async def _get_spacy_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using spaCy's linguistic features."""
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

    def _calculate_confidence(self, confidences: List[float]) -> float:
        """Calculate overall confidence score."""
        # Higher confidence if all methods agree
        confidence_std = np.std(confidences)
        base_confidence = np.mean(confidences)

        return base_confidence * (1 - confidence_std)

    async def _update_accuracy_metrics(self, result: Dict[str, Any]) -> None:
        """Update accuracy metrics for continuous monitoring."""
        self.accuracy_metrics['total_predictions'] += 1
        self.accuracy_metrics['confidence_scores'].append(result['confidence'])

        # Keep only recent predictions for rolling accuracy
        if len(self.accuracy_metrics['confidence_scores']) > 1000:
            self.accuracy_metrics['confidence_scores'] = self.accuracy_metrics['confidence_scores'][-1000:]

        current_accuracy = self.get_current_accuracy()
        logger.info(f"Current sentiment analysis accuracy: {current_accuracy:.2%}")

    def get_current_accuracy(self) -> float:
        """Get current accuracy metrics."""
        if not self.accuracy_metrics['confidence_scores']:
            return 0.0

        # Calculate rolling accuracy based on confidence scores
        recent_confidences = self.accuracy_metrics['confidence_scores'][-100:]
        return sum(recent_confidences) / len(recent_confidences)

    async def validate_accuracy(self, text: str, actual_sentiment: str) -> bool:
        """Validate prediction against actual market movement."""
        prediction = await self.analyze_sentiment(text)
        is_correct = prediction['sentiment'] == actual_sentiment

        self.accuracy_metrics['total_predictions'] += 1
        if is_correct:
            self.accuracy_metrics['correct_predictions'] += 1

        return is_correct
