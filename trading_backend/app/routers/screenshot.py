"""Screenshot analysis router."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict

from ..services.ocr.ocr_service import OCRService

router = APIRouter()
ocr_service = OCRService()

@router.post("/analyze")
async def analyze_screenshot(file: UploadFile = File(...)) -> Dict[str, str]:
    """Analyze screenshot using OCR."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        text = await ocr_service.extract_text(contents)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
