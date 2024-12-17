"""Test screenshot analysis functionality."""
import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import os

from app.main import app
from app.services.ocr.ocr_service import OCRService

@pytest.fixture
def test_client():
    return TestClient(app)

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
    # Create a white image using PIL
    img = Image.new('RGB', (300, 100), color='white')
    # Use PIL's ImageDraw to add Chinese text
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    # Use Noto Sans CJK font for Chinese characters
    try:
        # Try to use Noto Sans CJK SC (Simplified Chinese)
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 32)
    except:
        try:
            # Fallback to any available Noto CJK font
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 32)
        except:
            print("Warning: Could not load Noto CJK font, using default")
            font = ImageFont.load_default()

    # Add Chinese text with larger size and better positioning
    draw.text((50, 30), "你好世界", font=font, fill='black')
    # Save with high quality
    img.save(str(image_path), quality=95)
    return image_path

def test_ocr_service_initialization():
    """Test OCR service initialization."""
    service = OCRService()
    assert service is not None
    assert hasattr(service, 'extract_text')

@pytest.mark.asyncio
async def test_english_text_extraction(sample_image_path):
    """Test English text extraction from image."""
    service = OCRService()
    with open(sample_image_path, 'rb') as f:
        image_bytes = f.read()
    text = await service.extract_text(image_bytes)
    assert "Hello" in text
    assert "World" in text

@pytest.mark.asyncio
async def test_chinese_text_extraction(sample_chinese_image_path):
    """Test Chinese text extraction from image."""
    service = OCRService()
    with open(sample_chinese_image_path, 'rb') as f:
        image_bytes = f.read()
    text = await service.extract_text(image_bytes)
    assert "你好" in text
    assert "世界" in text

def test_screenshot_upload_endpoint(test_client, sample_image_path):
    """Test screenshot upload and analysis endpoint."""
    with open(sample_image_path, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        response = test_client.post("/api/screenshot/analyze", files=files)
    assert response.status_code == 200
    assert "text" in response.json()

def test_invalid_file_type(test_client):
    """Test invalid file type handling."""
    files = {'file': ('test.txt', io.BytesIO(b'not an image'), 'text/plain')}
    response = test_client.post("/api/screenshot/analyze", files=files)
    assert response.status_code == 400
