"""Test screenshot analysis functionality."""
import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os

from app.main import app
from app.services.ocr.ocr_service import OCRService

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def sample_trading_image_path(tmp_path):
    """Create a sample image with trading signals."""
    image_path = tmp_path / "test_trading.png"
    # Create a white image
    img = np.full((200, 600, 3), 255, dtype=np.uint8)
    # Add black text with trading signals
    font = cv2.FONT_HERSHEY_SIMPLEX
    signals = [
        "BTC showing bullish flag pattern",
        "Support level at $45,000",
        "Golden cross forming on 4h chart",
        "Higher triangle with accumulation"
    ]
    y_pos = 40
    for signal in signals:
        cv2.putText(img, signal, (10, y_pos), font, 0.7, (0, 0, 0), 2)
        y_pos += 40
    cv2.imwrite(str(image_path), img)
    return image_path

@pytest.fixture
def sample_chinese_trading_image_path(tmp_path):
    """Create a sample image with Chinese trading signals."""
    image_path = tmp_path / "test_chinese_trading.png"
    img = Image.new('RGB', (600, 200), color='white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 24)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 24)
        except:
            print("Warning: Could not load Noto CJK font, using default")
            font = ImageFont.load_default()

    signals = [
        "比特币形成多头旗形",
        "支撑位在45000",
        "即将出现黄金交叉",
        "上升三角形伴随筑底"
    ]
    y_pos = 30
    for signal in signals:
        draw.text((20, y_pos), signal, font=font, fill='black')
        y_pos += 40
    img.save(str(image_path), quality=95)
    return image_path

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image with English text for testing."""
    image_path = tmp_path / "test_image.png"
    # Create a white image
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    # Add black text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Hello World", (10, 50), font, 1, (0, 0, 0), 2)
    cv2.imwrite(str(image_path), img)
    return image_path

@pytest.fixture
def sample_chinese_image_path(tmp_path):
    """Create a sample image with Chinese text for testing."""
    image_path = tmp_path / "test_chinese.png"
    img = Image.new('RGB', (300, 100), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 32)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 32)
        except:
            print("Warning: Could not load Noto CJK font, using default")
            font = ImageFont.load_default()

    draw.text((50, 30), "你好世界", font=font, fill='black')
    img.save(str(image_path), quality=95)
    return image_path

def test_ocr_service_initialization():
    """Test OCR service initialization."""
    service = OCRService()
    assert service is not None
    assert hasattr(service, 'extract_text')
    assert hasattr(service, 'signal_extractor')

@pytest.mark.asyncio
async def test_trading_signal_extraction(sample_trading_image_path):
    """Test trading signal extraction from English image."""
    service = OCRService()
    with open(sample_trading_image_path, 'rb') as f:
        image_bytes = f.read()
    result = await service.extract_text(image_bytes)

    assert result['language'] == 'English'
    assert result['signals'] is not None
    assert result['signals']['sentiment'] == 'bullish'
    assert result['signals']['confidence'] > 0.7
    assert len(result['signals']['patterns']['bullish']) >= 2
    assert 45000 in result['signals']['prices']
    assert 'btc' in result['signals']['cryptocurrencies']

@pytest.mark.asyncio
async def test_chinese_trading_signal_extraction(sample_chinese_trading_image_path):
    """Test trading signal extraction from Chinese image."""
    service = OCRService()
    with open(sample_chinese_trading_image_path, 'rb') as f:
        image_bytes = f.read()
    result = await service.extract_text(image_bytes)

    assert result['language'] == 'Simplified Chinese'
    assert result['signals'] is not None
    assert result['signals']['sentiment'] == 'bullish'
    assert result['signals']['confidence'] > 0.7
    assert len(result['signals']['patterns']['bullish']) >= 2
    assert 45000 in result['signals']['prices']
    assert any('比特币' in crypto for crypto in result['signals']['cryptocurrencies'])

@pytest.mark.asyncio
async def test_english_text_extraction(sample_image_path):
    """Test English text extraction from image."""
    service = OCRService()
    with open(sample_image_path, 'rb') as f:
        image_bytes = f.read()
    result = await service.extract_text(image_bytes)
    assert "Hello" in result['text']
    assert "World" in result['text']

@pytest.mark.asyncio
async def test_chinese_text_extraction(sample_chinese_image_path):
    """Test Chinese text extraction from image."""
    service = OCRService()
    with open(sample_chinese_image_path, 'rb') as f:
        image_bytes = f.read()
    result = await service.extract_text(image_bytes)
    assert "你好" in result['text']
    assert "世界" in result['text']

def test_screenshot_upload_endpoint(test_client, sample_trading_image_path):
    """Test screenshot upload and analysis endpoint."""
    with open(sample_trading_image_path, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        response = test_client.post("/api/screenshot/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "signals" in data
    assert "confidence_explanation" in data

def test_invalid_file_type(test_client):
    """Test invalid file type handling."""
    files = {'file': ('test.txt', io.BytesIO(b'not an image'), 'text/plain')}
    response = test_client.post("/api/screenshot/analyze", files=files)
    assert response.status_code == 400

def test_signal_confidence_explanation(test_client, sample_trading_image_path):
    """Test confidence explanation generation."""
    with open(sample_trading_image_path, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        response = test_client.post("/api/screenshot/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "confidence_explanation" in data
    assert isinstance(data["confidence_explanation"], str)
    assert len(data["confidence_explanation"]) > 0
