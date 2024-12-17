import pytest
from app.services.notification.email_service import EmailNotificationService
from app.routers.notification import router
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import smtplib

@pytest.fixture
def email_service():
    return EmailNotificationService()

@pytest.fixture
def test_client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

def test_email_template_trading_signal(email_service):
    """Test trading signal email template generation."""
    template_params = {
        'pair': 'BTC/USDT',
        'entry_price': '50000',
        'take_profit': '55000',
        'stop_loss': '48000',
        'position_size': '1000',
        'confidence': '90',
        'fee': '1.5',
        'net_profit': '4998.5',
        'position_type': 'Full Position',
        'monitoring_note': 'Monitor for breakout confirmation',
        'timestamp': '2024-01-01 12:00:00'
    }

    content = email_service.get_template('trading_signal', **template_params)
    assert 'BTC/USDT' in content
    assert '50000' in content
    assert '90%' in content
    assert 'Full Position' in content

def test_email_template_position_update(email_service):
    """Test position update email template generation."""
    template_params = {
        'pair': 'ETH/USDT',
        'current_price': '3000',
        'entry_price': '2800',
        'pnl': '7.14',
        'stop_loss': '2700',
        'take_profit': '3300',
        'action': 'Consider taking partial profits'
    }

    content = email_service.get_template('position_update', **template_params)
    assert 'ETH/USDT' in content
    assert '3000' in content
    assert '7.14%' in content
    assert 'Consider taking partial profits' in content

@pytest.mark.asyncio
async def test_send_notification_success(email_service):
    """Test successful email notification sending."""
    with patch('smtplib.SMTP') as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        success = await email_service.send_notification(
            subject="Test Subject",
            content="Test Content",
            recipient="test@example.com"
        )

        assert success is True
        assert len(email_service.notification_queue) == 0
        mock_server.send_message.assert_called_once()

@pytest.mark.asyncio
async def test_send_notification_retry(email_service):
    """Test email notification retry mechanism."""
    with patch('smtplib.SMTP') as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        mock_server.send_message.side_effect = [
            smtplib.SMTPException("Test error"),
            None  # Success on second attempt
        ]

        success = await email_service.send_notification(
            subject="Test Subject",
            content="Test Content",
            recipient="test@example.com"
        )

        assert success is True
        assert len(email_service.failed_notifications) == 0
        assert mock_server.send_message.call_count == 2

def test_notification_endpoint_success(test_client):
    """Test notification endpoint with valid data."""
    with patch('app.services.notification.email_service.EmailNotificationService.send_notification') as mock_send:
        mock_send.return_value = True

        response = test_client.post("/api/notification/email", json={
            "template_type": "trading_signal",
            "recipient": "test@example.com",
            "params": {
                "pair": "BTC/USDT",
                "entry_price": "50000",
                "take_profit": "55000",
                "stop_loss": "48000",
                "position_size": "1000",
                "confidence": "90",
                "fee": "1.5",
                "net_profit": "4998.5",
                "position_type": "Full Position",
                "monitoring_note": "Monitor for breakout confirmation",
                "timestamp": "2024-01-01 12:00:00"
            }
        })

        assert response.status_code == 200
        assert response.json()["status"] == "success"

def test_notification_endpoint_invalid_data(test_client):
    """Test notification endpoint with invalid data."""
    response = test_client.post("/api/notification/email", json={
        "template_type": "trading_signal"
        # Missing required fields
    })

    assert response.status_code == 400
    assert "Missing required fields" in response.json()["detail"]

def test_queue_status_endpoint(test_client):
    """Test queue status endpoint."""
    response = test_client.get("/api/notification/queue")

    assert response.status_code == 200
    assert "queue_size" in response.json()
    assert "failed_notifications" in response.json()
