import os
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)

class EmailNotificationService:
    def __init__(self):
        """Initialize email notification service with SMTP configuration and queues."""
        self.smtp_config = {
            'server': 'smtp.qq.com',
            'port': 587,
            'email': '421940494@qq.com'
        }
        # Queue for pending notifications
        self.notification_queue = deque()
        # Track failed notifications for retry
        self.failed_notifications: List[Dict] = []
        # Maximum retry attempts
        self.max_retries = 3
        # Retry delay in seconds (starts at 1 minute)
        self.base_retry_delay = 60

    async def send_notification(self, subject: str, content: str, recipient: str) -> bool:
        """Send email notification with retry mechanism."""
        notification = {
            'subject': subject,
            'content': content,
            'recipient': recipient,
            'attempts': 0,
            'next_retry': None
        }

        # Add to queue
        self.notification_queue.append(notification)
        success = await self._process_notification(notification)

        # If failed and retries available, schedule retry
        if not success and notification['attempts'] < self.max_retries:
            return await self._retry_notification(notification)

        return success

    async def _retry_notification(self, notification: Dict) -> bool:
        """Handle notification retry with exponential backoff."""
        while notification['attempts'] < self.max_retries:
            delay = self.base_retry_delay * (2 ** (notification['attempts'] - 1))
            await asyncio.sleep(delay)

            success = await self._process_notification(notification)
            if success:
                return True

            notification['attempts'] += 1

        return False

    async def _process_notification(self, notification: Dict) -> bool:
        """Process a single notification."""
        try:
            message = self._create_email_message(
                notification['subject'],
                notification['content'],
                notification['recipient']
            )

            with smtplib.SMTP(self.smtp_config['server'], self.smtp_config['port']) as server:
                server.starttls()
                server.login(self.smtp_config['email'], os.getenv('EMAIL_PASSWORD'))
                server.send_message(message)

            # Success - remove from queue
            if notification in self.notification_queue:
                self.notification_queue.remove(notification)
            if notification in self.failed_notifications:
                self.failed_notifications.remove(notification)
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            if notification not in self.failed_notifications:
                self.failed_notifications.append(notification)
            return False

    def _create_email_message(self, subject: str, content: str, recipient: str) -> MIMEMultipart:
        """Create email message with proper formatting."""
        message = MIMEMultipart()
        message['From'] = self.smtp_config['email']
        message['To'] = recipient
        message['Subject'] = subject

        # Add content
        message.attach(MIMEText(content, 'plain'))
        return message

    async def process_queue(self) -> None:
        """Process all pending notifications in the queue."""
        while self.notification_queue:
            notification = self.notification_queue[0]
            await self._process_notification(notification)
            await asyncio.sleep(1)  # Prevent overwhelming the SMTP server

    def get_template(self, template_type: str, **kwargs) -> str:
        """Get email template based on signal type."""
        templates = {
            'trading_signal': """
Trading Signal Alert

Currency Pair: {pair}
Entry Price: {entry_price}
Take Profit: {take_profit}
Stop Loss: {stop_loss}
Position Size: {position_size} USDT
Confidence Level: {confidence}%
Transaction Fee: {fee} USDT
Net Profit Target: {net_profit} USDT

Position Type: {position_type}
{monitoring_note}

Signal generated at: {timestamp}
            """.strip(),

            'position_update': """
Position Update Alert

Currency Pair: {pair}
Current Price: {current_price}
Entry Price: {entry_price}
Current P/L: {pnl}%
Stop Loss: {stop_loss}
Take Profit: {take_profit}

Action Required: {action}
            """.strip()
        }

        template = templates.get(template_type)
        if not template:
            raise ValueError(f"Unknown template type: {template_type}")

        return template.format(**kwargs)
