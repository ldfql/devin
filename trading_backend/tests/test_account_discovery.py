"""Tests for the account discovery service."""
import pytest
from datetime import datetime, timedelta
from app.services.web_scraping.account_discovery import AccountDiscoveryService

@pytest.fixture
async def discovery_service():
    service = AccountDiscoveryService()
    await service.initialize()
    return service

@pytest.mark.asyncio
async def test_initialization(discovery_service):
    """Test service initialization."""
    assert len(discovery_service.seed_accounts['twitter']) > 0
    assert len(discovery_service.seed_accounts['youtube']) > 0

@pytest.mark.asyncio
async def test_rate_limiting(discovery_service):
    """Test rate limiting functionality."""
    platform = 'twitter'

    # First update should be allowed
    assert discovery_service._can_update(platform) is True

    # Immediate second update should be blocked
    assert discovery_service._can_update(platform) is False

@pytest.mark.asyncio
async def test_add_seed_account(discovery_service):
    """Test adding new seed accounts."""
    new_account = 'test_account'
    platform = 'twitter'

    # Add new account
    result = discovery_service.add_seed_account(platform, new_account)
    assert result is True
    assert new_account in discovery_service.seed_accounts[platform]

    # Try adding same account again
    result = discovery_service.add_seed_account(platform, new_account)
    assert result is False

@pytest.mark.asyncio
async def test_get_top_accounts(discovery_service):
    """Test getting top accounts."""
    platform = 'twitter'
    discovery_service.discovered_accounts[platform] = {
        'account1': {'engagement_rate': 0.8, 'followers': 1000},
        'account2': {'engagement_rate': 0.9, 'followers': 2000},
        'account3': {'engagement_rate': 0.7, 'followers': 500}
    }

    top_accounts = discovery_service.get_top_accounts(platform, limit=2)
    assert len(top_accounts) == 2
    assert top_accounts[0] == 'account2'  # Highest engagement rate
    assert top_accounts[1] == 'account1'  # Second highest

@pytest.mark.asyncio
async def test_get_account_metrics(discovery_service):
    """Test getting account metrics."""
    platform = 'twitter'
    account = 'test_account'
    metrics = {'engagement_rate': 0.8, 'followers': 1000}

    discovery_service.discovered_accounts[platform] = {account: metrics}

    result = discovery_service.get_account_metrics(platform, account)
    assert result == metrics

    # Test non-existent account
    result = discovery_service.get_account_metrics(platform, 'non_existent')
    assert result is None
