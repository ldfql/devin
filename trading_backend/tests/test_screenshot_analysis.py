import pytest
import asyncio
from pathlib import Path
from app.services.ocr.signal_extractor import SignalExtractor
from app.services.ocr.ocr_service import OCRService

@pytest.fixture
async def signal_extractor():
    return SignalExtractor()

@pytest.fixture
async def ocr_service():
    return OCRService()

@pytest.mark.asyncio
async def test_english_screenshot_analysis(signal_extractor):
    """Test analysis of English trading signals from screenshots."""
    # Load test data
    test_file = Path(__file__).parent / "data/test_screenshot_en.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = f.read()

    # Analyze signals
    signals = await signal_extractor.extract_signals(test_data)

    assert len(signals) > 0, "Should extract trading signals"
    for signal in signals:
        assert signal["confidence"] >= 0.85, "Signal confidence below threshold"
        assert "price_target" in signal
        assert "stop_loss" in signal
        assert "market_direction" in signal

@pytest.mark.asyncio
async def test_chinese_screenshot_analysis(signal_extractor):
    """Test analysis of Chinese trading signals from screenshots."""
    # Load test data
    test_file = Path(__file__).parent / "data/test_screenshot_cn.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = f.read()

    # Analyze signals
    signals = await signal_extractor.extract_signals(test_data)

    assert len(signals) > 0, "Should extract trading signals"
    for signal in signals:
        assert signal["confidence"] >= 0.85, "Signal confidence below threshold"
        assert "price_target" in signal
        assert "stop_loss" in signal
        assert "market_direction" in signal

@pytest.mark.asyncio
async def test_ocr_accuracy(ocr_service):
    """Test OCR accuracy for both languages."""
    # Test English OCR
    en_test_file = Path(__file__).parent / "data/test_screenshot_en.txt"
    with open(en_test_file, "r", encoding="utf-8") as f:
        en_test_data = f.read()

    en_result = await ocr_service.extract_text(en_test_data)
    assert en_result["accuracy"] >= 0.85, "English OCR accuracy below threshold"

    # Test Chinese OCR
    cn_test_file = Path(__file__).parent / "data/test_screenshot_cn.txt"
    with open(cn_test_file, "r", encoding="utf-8") as f:
        cn_test_data = f.read()

    cn_result = await ocr_service.extract_text(cn_test_data)
    assert cn_result["accuracy"] >= 0.85, "Chinese OCR accuracy below threshold"

@pytest.mark.asyncio
async def test_technical_pattern_recognition(signal_extractor):
    """Test recognition of technical patterns in screenshots."""
    patterns = [
        "double_top",
        "double_bottom",
        "head_and_shoulders",
        "triangle",
        "wedge"
    ]

    for pattern in patterns:
        test_file = Path(__file__).parent / f"data/test_pattern_{pattern}.txt"
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = f.read()

        result = await signal_extractor.detect_pattern(test_data)
        assert result["pattern"] == pattern, f"Failed to detect {pattern} pattern"
        assert result["confidence"] >= 0.85, f"Pattern detection confidence below threshold for {pattern}"

@pytest.mark.asyncio
async def test_multi_language_pattern_description(signal_extractor):
    """Test pattern description in multiple languages."""
    test_file = Path(__file__).parent / "data/test_pattern_complex.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = f.read()

    # Test English description
    en_result = await signal_extractor.analyze_pattern(test_data, language="en")
    assert en_result["description"], "Missing English pattern description"
    assert en_result["confidence"] >= 0.85, "English analysis confidence below threshold"

    # Test Chinese description
    cn_result = await signal_extractor.analyze_pattern(test_data, language="zh")
    assert cn_result["description"], "Missing Chinese pattern description"
    assert cn_result["confidence"] >= 0.85, "Chinese analysis confidence below threshold"

@pytest.mark.asyncio
async def test_signal_validation(signal_extractor):
    """Test validation of extracted trading signals."""
    test_file = Path(__file__).parent / "data/test_screenshot_mixed.txt"
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = f.read()

    signals = await signal_extractor.extract_signals(test_data)

    for signal in signals:
        validation = await signal_extractor.validate_signal(signal)
        assert validation["is_valid"], f"Signal validation failed: {validation['reason']}"
        assert validation["confidence"] >= 0.85, "Signal validation confidence below threshold"
