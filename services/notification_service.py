from utils.socket_manager import sio
from utils.logger import logger
from typing import Dict, Any
from datetime import datetime


class NotificationService:
    @staticmethod
    async def send_user_notification(user_id: str, hr_id: str, notification_data: Dict[str, Any]):
        """
        Send notification to a specific user
        """
        try:
            # Ensure timestamp is included
            notification_data['timestamp'] = datetime.now().isoformat()
            
            # Create the full notification payload
            full_notification = {
                'type': 'user_notification',
                'user_id': user_id,
                'hr_id': hr_id,
                'data': notification_data,
                'timestamp': notification_data['timestamp']
            }
            
            # Emit to the user's specific room
            room_name = f"user_{user_id}"
            await sio.emit('notification', full_notification, room=room_name)
            await sio.emit('hr_notification', full_notification, room=f"user_{hr_id}")
        
            
        except Exception as e:
            logger.error(f"Error sending notification to user {user_id}: {str(e)}")
            raise

    @staticmethod
    async def send_channel_notification(channel: str, notification_data: Dict[str, Any]):
        """
        Send notification to a channel (all subscribers)
        """
        try:
            notification_data['timestamp'] = datetime.now().isoformat()
            
            await sio.emit('notification', {
                'type': 'channel_notification',
                'channel': channel,
                'data': notification_data,
                'timestamp': notification_data['timestamp']
            }, room=f"channel_{channel}")
            

            
        except Exception as e:
            logger.error(f"Error sending notification to channel {channel}: {str(e)}")
            raise

    @staticmethod
    async def send_broadcast_notification(notification_data: Dict[str, Any]):
        """
        Send notification to all connected clients
        """
        try:
            notification_data['timestamp'] = datetime.now().isoformat()
            
            await sio.emit('notification', {
                'type': 'broadcast',
                'data': notification_data,
                'timestamp': notification_data['timestamp']
            })
            
        
            
        except Exception as e:
            logger.error(f"Error sending broadcast notification: {str(e)}")
            raise