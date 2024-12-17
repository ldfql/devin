import os
import re
import logging
import spacy
import torch
import numpy as np
from typing import Dict, Any, List
from functools import lru_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnglishSentimentAnalyzer:
    """Advanced sentiment analyzer for English text with high accuracy requirements."""

    # Strong positive patterns with clear bullish signals
    STRONG_POSITIVE_PATTERNS = [
        r'\bstrong buy\b', r'\bbreakout\b', r'\bmassive gains\b', r'\bmooning\b',
        r'\ball-time high\b', r'\bparabolic\b', r'\baccumulation phase\b',
        r'\bhighly bullish\b', r'\bextremely bullish\b', r'\bmassive volume\b',
        r'\bgolden cross\b', r'\bprice discovery\b', r'\bsqueeze\b', r'\bfomo\b',
        r'\bhodl\b', r'\bto the moon\b', r'\bdiamond hands\b', r'\bbtd\b'
    ]

    # Strong negative patterns with clear bearish signals
    STRONG_NEGATIVE_PATTERNS = [
        r'\bstrong sell\b', r'\bbreakdown\b', r'\bmassive losses\b', r'\bcrashing\b',
        r'\ball-time low\b', r'\bcapitulation\b', r'\bdistribution phase\b',
        r'\bhighly bearish\b', r'\bextremely bearish\b', r'\bpanic selling\b',
        r'\bdeath cross\b', r'\bmarket dump\b', r'\bsell signal\b', r'\bstop loss\b',
        r'\bpaper hands\b', r'\brug pull\b', r'\bexit scam\b', r'\bwhale dump\b'
    ]

    # Regular positive patterns
    POSITIVE_PATTERNS = [
        r'\bbullish\b', r'\buptrend\b', r'\bsupport\b', r'\bbounce\b', r'\brally\b',
        r'\blong\b', r'\bbuying\b', r'\baccumulate\b', r'\brecovery\b', r'\bgrowth\b',
        r'\bpositive\b', r'\boptimistic\b', r'\bconfidence\b', r'\bstrong\b',
        r'\bbreaking resistance\b', r'\bhigher high\b', r'\bhigher low\b',
        r'\bhold\b', r'\bbuy signal\b', r'\bstrong volume\b', r'\bprice target\b',
        r'\bconfirming\b', r'\bcontinuation\b', r'\bwhale accumulation\b'
    ]

    # Regular negative patterns
    NEGATIVE_PATTERNS = [
        r'\bbearish\b', r'\bdowntrend\b', r'\bresistance\b', r'\bdrop\b', r'\bsell-off\b',
        r'\bshort\b', r'\bselling\b', r'\bdistribute\b', r'\bdecline\b', r'\bweakness\b',
        r'\bnegative\b', r'\bpessimistic\b', r'\bfear\b', r'\bweak\b',
        r'\bbreaking support\b', r'\blower low\b', r'\blower high\b',
        r'\bexit\b', r'\bweak bounce\b', r'\bselling pressure\b',
        r'\btest support\b', r'\breversal\b', r'\bbreaking down\b'
    ]

    def __init__(self):
        """Initialize the sentiment analyzer."""
        self.initialized = False
        self.finbert = None
        self.tokenizer = None
        self.nlp = None

        # Initialize accuracy metrics
        self.accuracy_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'bullish_accuracy': 0,
            'bearish_accuracy': 0,
            'neutral_accuracy': 0,
            'last_100_predictions': [],  # Store last 100 prediction results
            'rolling_accuracy': 0.0      # Rolling accuracy over last 100 predictions
        }

        # Component weights for ensemble
        self.weights = {
            'finbert': 0.6,
            'pattern': 0.3,
            'spacy': 0.1
        }

    async def initialize(self):
        """Initialize models and resources."""
        try:
            if not self.initialized:
                # Initialize spaCy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("Default spaCy model not found. Using blank model.")
                    self.nlp = spacy.blank("en")

                # Initialize FinBERT
                model_name = "ProsusAI/finbert"
                logger.info(f"Loading FinBERT model: {model_name}")

                # Load tokenizer and model with proper configuration
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.finbert = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=3,  # positive, negative, neutral
                    output_attentions=False,
                    output_hidden_states=False
                )

                # Move model to evaluation mode
                self.finbert.eval()

                # Initialize accuracy tracking
                self.accuracy_metrics = {
                    'total_predictions': 0,
                    'correct_predictions': 0,
                    'bullish_accuracy': 0,
                    'bearish_accuracy': 0,
                    'neutral_accuracy': 0,
                    'last_100_predictions': [],
                    'rolling_accuracy': 0.0
                }

                self.initialized = True
                logger.info("Sentiment analyzer initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of English text."""
        try:
            # Get sentiment from different components
            finbert_result = await self._get_finbert_sentiment(text)
            pattern_result = await self._get_pattern_sentiment(text)
            spacy_result = self._get_spacy_sentiment(text)

            logger.info(f"FinBERT sentiment: {finbert_result}")
            logger.info(f"Pattern sentiment: {pattern_result}")
            logger.info(f"spaCy sentiment: {spacy_result}")

            # Calculate weighted ensemble score with adjusted weights
            finbert_weight = 0.4
            pattern_weight = 0.5
            spacy_weight = 0.1

            ensemble_score = (
                finbert_result['score'] * finbert_weight +
                pattern_result['score'] * pattern_weight +
                spacy_result['score'] * spacy_weight
            )

            # Check for strong technical patterns first
            strong_technical = (
                pattern_result['matches']['strong_positive'] +
                pattern_result['matches']['strong_negative']
            ) >= 2

            # Enhanced neutral signal detection with stricter criteria
            neutral_signals = [
                finbert_result['probabilities']['neutral'] > 0.7,
                pattern_result['matches'].get('neutral', 0) > 1,
                abs(ensemble_score) < 0.15,
                abs(pattern_result['score']) < 0.2,
                abs(finbert_result['score']) < 0.3
            ]

            # Calculate base confidence
            base_confidence = (
                finbert_result['confidence'] * finbert_weight +
                pattern_result['confidence'] * pattern_weight +
                spacy_result['confidence'] * spacy_weight
            ) * 1.25

            # Add agreement bonus with higher weight for technical patterns
            agreement_bonus = self._calculate_agreement_bonus([
                finbert_result['score'],
                pattern_result['score'] * 1.5,
                spacy_result['score']
            ]) * 1.2

            confidence = min(0.95, base_confidence + agreement_bonus)

            # Determine sentiment with priority on technical patterns
            if strong_technical:
                sentiment = 'bullish' if pattern_result['score'] > 0 else 'bearish'
                confidence = min(0.95, confidence * 1.25)
            # Strong pattern signal with support
            elif abs(pattern_result['score']) > 0.6 and (
                pattern_result['score'] * finbert_result['score'] > 0 or
                pattern_result['matches']['strong_positive'] + pattern_result['matches']['strong_negative'] >= 1
            ):
                sentiment = 'bullish' if pattern_result['score'] > 0 else 'bearish'
                confidence = min(0.95, confidence * 1.2)
            # Strong FinBERT signal with pattern support
            elif abs(finbert_result['score']) > 0.7 and (
                finbert_result['score'] * pattern_result['score'] > 0 or
                finbert_result['confidence'] > 0.9
            ):
                sentiment = 'bullish' if finbert_result['score'] > 0 else 'bearish'
                confidence = min(0.95, confidence * 1.15)
            # Clear neutral signal
            elif sum(neutral_signals) >= 4:
                sentiment = 'neutral'
                confidence = min(0.8, confidence * 0.9)
            # Strong ensemble signal
            elif abs(ensemble_score) > 0.3:
                sentiment = 'bullish' if ensemble_score > 0 else 'bearish'
                confidence = min(0.95, confidence * 1.1)
            else:
                sentiment = 'neutral'
                confidence = min(0.75, confidence * 0.85)

            result = {
                'sentiment': sentiment,
                'confidence': confidence,
                'score': ensemble_score,
                'components': {
                    'finbert': finbert_result,
                    'pattern': pattern_result,
                    'spacy': spacy_result
                }
            }

            # Update accuracy metrics
            await self._update_accuracy_metrics(result)

            return result

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise

    async def _get_finbert_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using FinBERT model."""
        try:
            # Tokenize and get model outputs
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get probabilities for each class
            probs = probabilities[0].detach().numpy()
            sentiment_probs = {
                'positive': float(probs[0]),
                'negative': float(probs[1]),
                'neutral': float(probs[2])
            }

            # Calculate sentiment score (-1 to 1)
            # Positive probability contributes positive, negative probability contributes negative
            score = sentiment_probs['positive'] - sentiment_probs['negative']

            # Calculate confidence based on probability distribution
            max_prob = max(sentiment_probs.values())
            second_max_prob = sorted(sentiment_probs.values())[-2]

            # Higher confidence if there's a clear winner
            confidence_margin = max_prob - second_max_prob
            base_confidence = max_prob * 0.7 + confidence_margin * 0.3

            # Boost confidence for strong signals
            if max_prob > 0.8 and confidence_margin > 0.3:
                confidence = min(0.95, base_confidence * 1.2)
            else:
                confidence = min(0.9, base_confidence)

            logger.info(f"FinBERT probabilities: {sentiment_probs}")
            logger.info(f"FinBERT score: {score:.2f}, confidence: {confidence:.2f}")

            return {
                'score': score,
                'confidence': confidence,
                'probabilities': sentiment_probs
            }

        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {str(e)}")
            raise

    async def _get_pattern_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment based on pattern matching."""
        try:
            # Strong positive patterns (cryptocurrency-specific and technical analysis)
            strong_positive = [
                r'bull(?:ish|run)|golden\s*cross|breakout|accumulation',
                r'support.{1,20}hold|higher.{1,20}high',
                r'institutional.{1,20}(buy|invest|accumulate)',
                r'strong.{1,20}(buy|support|momentum)',
                r'surge.{1,20}\d+%|\d+%.{1,20}(gain|up)',
                r'(buy|long).{1,20}pressure|volume.{1,20}spike',
                r'key\s*resistance.{1,20}break|resistance.{1,20}cleared',
                r'bull(?:ish)?.{1,20}flag|cup.{1,20}handle',
                r'double\s*bottom|inverse\s*head.{1,20}shoulder',
                r'(higher|ascending).{1,20}triangle',
                r'whale.{1,20}(buy|accumulate)|smart\s*money.{1,20}in',
                r'short.{1,20}squeeze|fomo',
                r'oversold.{1,20}bounce|reversal.{1,20}bottom'
            ]

            # Strong negative patterns (expanded technical patterns)
            strong_negative = [
                r'bear(?:ish|market)|death\s*cross|breakdown',
                r'resistance.{1,20}reject|lower.{1,20}low',
                r'(sell|short).{1,20}pressure|dump',
                r'support.{1,20}break|fail.{1,20}support',
                r'crash.{1,20}\d+%|\d+%.{1,20}(loss|down)',
                r'liquidation|capitulation|panic.{1,20}sell',
                r'bearish.{1,20}divergence|distribution',
                r'bear(?:ish)?.{1,20}flag|head.{1,20}shoulder',
                r'double\s*top|(descending|falling).{1,20}triangle',
                r'whale.{1,20}(dump|sell)|smart\s*money.{1,20}out',
                r'long.{1,20}liquidation|stop.{1,20}hunt',
                r'overbought.{1,20}reject|reversal.{1,20}top'
            ]

            # Neutral patterns (expanded with technical terms)
            neutral = [
                r'consolidat|range.{1,20}bound|sideways',
                r'stable|steady|balanced',
                r'await|hold|pause',
                r'volume.{1,20}(decrease|low)',
                r'indecisive|unclear|mixed',
                r'tight.{1,20}range|no.{1,20}clear.{1,20}direction',
                r'neutral|equilibrium|balance',
                r'inside.{1,20}bar|doji',
                r'accumulation.{1,20}distribution|chop',
                r'low.{1,20}volatility|flat.{1,20}trend'
            ]

            # Positive but not strong patterns
            positive = [
                r'uptrend|higher|rise|gain',
                r'buy|long|support',
                r'confidence|optimistic|promising',
                r'recover|rebound|bounce',
                r'dip.{1,20}buy|accumulate',
                r'trend.{1,20}line.{1,20}hold|support.{1,20}test',
                r'higher.{1,20}timeframe|bullish.{1,20}bias'
            ]

            # Negative but not strong patterns
            negative = [
                r'downtrend|lower|fall|drop',
                r'sell|short|resistance',
                r'weakness|pessimistic|concerning',
                r'decline|pullback|correction',
                r'profit.{1,20}taking|sell.{1,20}pressure',
                r'trend.{1,20}line.{1,20}break|resistance.{1,20}test',
                r'lower.{1,20}timeframe|bearish.{1,20}bias'
            ]

            # Count pattern matches with improved regex
            text_lower = text.lower()
            strong_positive_matches = sum(len(re.findall(pattern, text_lower)) for pattern in strong_positive)
            strong_negative_matches = sum(len(re.findall(pattern, text_lower)) for pattern in strong_negative)
            positive_matches = sum(len(re.findall(pattern, text_lower)) for pattern in positive)
            negative_matches = sum(len(re.findall(pattern, text_lower)) for pattern in negative)
            neutral_matches = sum(len(re.findall(pattern, text_lower)) for pattern in neutral)

            # Calculate weighted score with adjusted weights
            total_matches = (
                strong_positive_matches * 3.0 +  # Increased weight for strong signals
                positive_matches +
                -strong_negative_matches * 3.0 -  # Increased weight for strong signals
                negative_matches
            )

            # Normalize score to [-1, 1] range with emphasis on strong signals
            total_weight = (
                strong_positive_matches * 3.0 +
                positive_matches +
                strong_negative_matches * 3.0 +
                negative_matches +
                neutral_matches
            )

            if total_weight == 0:
                score = 0
                confidence = 0.3
            else:
                score = total_matches / (total_weight * 1.5)
                # Calculate confidence with higher emphasis on strong patterns
                strong_matches = strong_positive_matches + strong_negative_matches
                confidence = min(0.95, (
                    strong_matches * 0.4 +  # Increased weight for strong matches
                    (positive_matches + negative_matches) * 0.15 +
                    neutral_matches * 0.1  # Reduced weight for neutral matches
                ))

            return {
                'score': max(-1, min(1, score)),
                'confidence': confidence,
                'matches': {
                    'strong_positive': strong_positive_matches,
                    'strong_negative': strong_negative_matches,
                    'positive': positive_matches,
                    'negative': negative_matches,
                    'neutral': neutral_matches
                }
            }

        except Exception as e:
            logger.error(f"Error in pattern sentiment analysis: {str(e)}")
            return {'score': 0, 'confidence': 0.1, 'matches': {}}

    def _get_spacy_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using spaCy's built-in sentiment analysis."""
        try:
            doc = self.nlp(text)

            # Financial terms that modify sentiment
            financial_modifiers = {
                'positive': ['support', 'buy', 'long', 'bullish', 'uptrend', 'profit'],
                'negative': ['resistance', 'sell', 'short', 'bearish', 'downtrend', 'loss']
            }

            # Calculate base sentiment
            tokens = [token.text.lower() for token in doc]
            positive_count = sum(1 for token in tokens if token in financial_modifiers['positive'])
            negative_count = sum(1 for token in tokens if token in financial_modifiers['negative'])

            total_count = positive_count + negative_count
            if total_count == 0:
                score = 0.0
                confidence = 0.1
            else:
                score = (positive_count - negative_count) / total_count
                # Higher confidence with more financial terms
                confidence = min(0.8, 0.3 + (total_count * 0.1))

            return {
                'score': max(-1.0, min(1.0, score)),
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error in spaCy sentiment analysis: {str(e)}")
            raise

    def _calculate_confidence(self, confidences: List[float]) -> float:
        """Calculate overall confidence based on individual component confidences."""
        if not confidences:
            return 0.0

        # Weight higher confidences more heavily
        sorted_confidences = sorted(confidences, reverse=True)
        weighted_sum = sum(conf * (0.5 ** i) for i, conf in enumerate(sorted_confidences))
        max_possible = sum(0.5 ** i for i in range(len(confidences)))

        return min(0.95, weighted_sum / max_possible)

    def _calculate_agreement_bonus(self, scores: List[float]) -> float:
        """Calculate confidence bonus based on agreement between components."""
        # Count components agreeing on direction
        signs = [1 if score > 0 else -1 if score < 0 else 0 for score in scores]
        positive_count = sum(1 for s in signs if s > 0)
        negative_count = sum(1 for s in signs if s < 0)
        neutral_count = sum(1 for s in signs if s == 0)

        # Calculate magnitude of agreement
        max_count = max(positive_count, negative_count, neutral_count)
        total_magnitude = sum(abs(score) for score in scores) / len(scores)

        # Strong agreement bonus when all components agree
        if max_count == len(scores):
            return min(0.3, 0.25 + total_magnitude * 0.1)
        # Moderate agreement bonus when majority agrees
        elif max_count >= 2:
            return min(0.25, 0.15 + total_magnitude * 0.15)
        # Small bonus for partial agreement with strong signals
        elif total_magnitude > 0.5:
            return 0.1

        return 0.05  # Minimal bonus for weak agreement

    async def _update_accuracy_metrics(self, result: Dict[str, Any]) -> None:
        """Update accuracy metrics based on prediction results."""
        try:
            # Update last 100 predictions list
            if len(self.accuracy_metrics['last_100_predictions']) >= 100:
                self.accuracy_metrics['last_100_predictions'].pop(0)
            self.accuracy_metrics['last_100_predictions'].append({
                'sentiment': result['sentiment'],
                'confidence': result['confidence']
            })

            # Update total predictions
            self.accuracy_metrics['total_predictions'] += 1

            logger.info(f"Updated accuracy metrics - Total predictions: {self.accuracy_metrics['total_predictions']}")

        except Exception as e:
            logger.error(f"Error updating accuracy metrics: {str(e)}")
            raise

    def get_current_accuracy(self) -> float:
        """Get the current accuracy based on validated predictions."""
        try:
            if self.accuracy_metrics['total_predictions'] == 0:
                return 0.0
            return self.accuracy_metrics['correct_predictions'] / self.accuracy_metrics['total_predictions']
        except Exception as e:
            logger.error(f"Error calculating current accuracy: {str(e)}")
            return 0.0

    async def validate_accuracy(self, text: str, expected_sentiment: str) -> bool:
        """Validate the accuracy of sentiment prediction."""
        try:
            result = await self.analyze_sentiment(text)
            is_correct = result['sentiment'] == expected_sentiment

            # Update accuracy metrics
            if is_correct:
                self.accuracy_metrics['correct_predictions'] += 1

            # Update sentiment-specific accuracy
            sentiment_key = f"{expected_sentiment}_accuracy"
            if sentiment_key in self.accuracy_metrics:
                self.accuracy_metrics[sentiment_key] = (
                    self.accuracy_metrics['correct_predictions'] /
                    self.accuracy_metrics['total_predictions']
                )

            logger.info(f"Validation - Expected: {expected_sentiment}, Got: {result['sentiment']}, Correct: {is_correct}")
            logger.info(f"Current accuracy: {self.get_current_accuracy():.2%}")

            return is_correct

        except Exception as e:
            logger.error(f"Error validating accuracy: {str(e)}")
            raise
