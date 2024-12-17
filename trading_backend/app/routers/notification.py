from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.notification.email_service import EmailNotificationService
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/notification", tags=["notification"])
email_service = EmailNotificationService()

@router.post("/email")
async def send_email_notification(data: Dict[str, Any]):
    """Send email notification with trading signal."""
    try:
        # Validate required fields
        required_fields = ['template_type', 'recipient', 'params']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )

        try:
            # Get email content from template
            content = email_service.get_template(
                template_type=data['template_type'],
                **data['params']
            )
        except (KeyError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid template data: {str(e)}"
            )

        # Send notification
        subject = f"Trading Alert: {data['params'].get('pair', 'Unknown')}"
        success = await email_service.send_notification(
            subject=subject,
            content=content,
            recipient=data['recipient']
        )

        if success:
            return {"status": "success", "message": "Notification sent successfully"}
        else:
            return {"status": "queued", "message": "Notification queued for retry"}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error sending notification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue")
async def get_queue_status():
    """Get status of notification queue."""
    return {
        "queue_size": len(email_service.notification_queue),
        "failed_notifications": len(email_service.failed_notifications)
    }
