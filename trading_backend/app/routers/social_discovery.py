from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
from ..services.web_scraping.account_discovery import AccountDiscoveryService

router = APIRouter(prefix="/api/social-discovery", tags=["social-discovery"])

@router.get("/experts")
async def discover_experts() -> List[Dict[str, str]]:
    """Discover cryptocurrency trading experts across social platforms."""
    try:
        discovery_service = AccountDiscoveryService()
        experts = await discovery_service.find_experts()
        return experts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/institutions")
async def discover_institutions() -> List[Dict[str, str]]:
    """Discover financial institutions across social platforms."""
    try:
        discovery_service = AccountDiscoveryService()
        institutions = await discovery_service.find_institutions()
        return institutions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending")
async def get_trending_accounts() -> List[Dict[str, str]]:
    """Get trending cryptocurrency-related accounts."""
    try:
        discovery_service = AccountDiscoveryService()
        trending = await discovery_service.get_trending_accounts()
        return trending
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
