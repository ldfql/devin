"""OCR service for processing screenshots."""
import os
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from typing import Dict, Optional

from .signal_extractor import SignalExtractor

logger = logging.getLogger(__name__)

class OCRService:
    """Service for performing OCR on screenshots."""

    def __init__(self):
        """Initialize OCR service with language support."""
        self.supported_languages = {
            'eng': 'English',
            'chi_sim': 'Chinese'
        }
        # Configure Tesseract
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Configure Tesseract parameters
        self.custom_config = r'--oem 1 --psm 3'

        # Initialize signal extractor
        self.signal_extractor = SignalExtractor()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        try:
            # Save original image for debugging
            cv2.imwrite('/tmp/debug_original.png', image)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('/tmp/debug_gray.png', gray)

            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast = clahe.apply(gray)
            cv2.imwrite('/tmp/debug_contrast.png', contrast)

            # Scale up image for better character recognition
            height, width = contrast.shape
            scaled = cv2.resize(contrast, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('/tmp/debug_scaled.png', scaled)

            # Denoise with minimal strength
            denoised = cv2.fastNlMeansDenoising(scaled, None, 5, 7, 21)
            cv2.imwrite('/tmp/debug_denoised.png', denoised)

            # Adaptive thresholding with parameters tuned for Chinese characters
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 15
            )
            cv2.imwrite('/tmp/debug_binary.png', binary)

            logger.info("Image preprocessing completed, debug images saved to /tmp/")
            return binary
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def detect_language(self, image: np.ndarray) -> str:
        """Detect language by trying both English and Chinese OCR."""
        # Try Chinese first
        chi_text = pytesseract.image_to_string(
            image, lang='chi_sim', config=self.custom_config
        ).strip()

        # If Chinese characters detected, use Chinese
        if any('\u4e00' <= char <= '\u9fff' for char in chi_text):
            return 'chi_sim'
        return 'eng'

    async def extract_text(self, image_bytes: bytes) -> Dict:
        """Extract text and trading signals from image bytes."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image bytes")

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Detect language
            lang = self.detect_language(processed_image)
            logger.info(f"Detected language: {lang}")

            # Extract text with detected language
            text = pytesseract.image_to_string(
                processed_image, lang=lang, config=self.custom_config
            ).strip()

            logger.info(f"Extracted text: {text}")

            # Extract trading signals
            signals = self.signal_extractor.extract_signals(
                text,
                'en' if lang == 'eng' else 'cn'
            )

            return {
                'text': text,
                'language': self.supported_languages[lang],
                'signals': signals,
                'confidence': signals.get('confidence', 0.0) if signals else 0.0,
                'sentiment': signals.get('sentiment', 'neutral') if signals else 'neutral'
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                'text': "",
                'language': "Unknown",
                'signals': None,
                'confidence': 0.0,
                'sentiment': 'neutral'
            }

    async def extract_text_from_string(self, text: str) -> Dict:
        """Extract trading signals from text string."""
        try:
            # Detect language based on text content
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                lang = 'chi_sim'
            else:
                lang = 'eng'

            # Extract trading signals
            signals = self.signal_extractor.extract_signals(
                text,
                'en' if lang == 'eng' else 'cn'
            )

            return {
                'text': text,
                'language': self.supported_languages[lang],
                'signals': signals,
                'confidence': signals.get('confidence', 0.0) if signals else 0.0,
                'sentiment': signals.get('sentiment', 'neutral') if signals else 'neutral'
            }

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return {
                'text': text,
                'language': "Unknown",
                'signals': None,
                'confidence': 0.0,
                'sentiment': 'neutral'
            }
