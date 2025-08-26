# utils/notifications.py
from utils.socket_manager import sio
from utils.logger import logger
from typing import Dict, Any
from datetime import datetime

class NotificationService:
    @staticmethod
    async def send_user_notification(user_id: str, notification_data: Dict[str, Any]):
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
                'data': notification_data,
                'timestamp': notification_data['timestamp']
            }
            
            # print(f"📤 Sending FULL notification to user {user_id}:")
            # print(f"   Room: user_{user_id}")
            # print(f"   Payload: {full_notification}")
            
            # Get connected clients in the room for debugging
            room_name = f"user_{user_id}"
            if hasattr(sio, 'manager') and room_name in sio.manager.rooms:
                clients_in_room = len(sio.manager.rooms[room_name])
                # print(f"   Clients in room '{room_name}': {clients_in_room}")
            else:
                print(f"   Room '{room_name}' does not exist or has no clients")
            
            # Emit to the user's specific room
            await sio.emit('notification', full_notification, room=room_name)

            # logger.info(f"Notification sent to user {user_id}: {notification_data.get('message')}")
            
        except Exception as e:
            logger.error(f"Error sending notification to user {user_id}: {str(e)}")
            print(f"❌ Error sending notification: {e}")

    @staticmethod
    async def send_channel_notification(channel: str, notification_data: Dict[str, Any]):
        """
        Send notification to a channel (all subscribers)
        """
        try:
            notification_data['timestamp'] = datetime.now().isoformat()
            
            # print(f"📤 Sending notification to channel {channel}: {notification_data['message']}")
            
            await sio.emit('notification', {
                'type': 'channel_notification',
                'channel': channel,
                'data': notification_data,
                'timestamp': notification_data['timestamp']
            }, room=f"channel_{channel}")
            
            # logger.info(f"Notification sent to channel {channel}: {notification_data.get('message')}")
            
        except Exception as e:
            logger.error(f"Error sending notification to channel {channel}: {str(e)}")
            print(f"❌ Error sending channel notification: {e}")

    @staticmethod
    async def send_broadcast_notification(notification_data: Dict[str, Any]):
        """
        Send notification to all connected clients
        """
        try:
            notification_data['timestamp'] = datetime.now().isoformat()
            
            # print(f"📤 Broadcasting notification: {notification_data['message']}")
            
            await sio.emit('notification', {
                'type': 'broadcast',
                'data': notification_data,
                'timestamp': notification_data['timestamp']
            })
            
            # logger.info(f"Broadcast notification: {notification_data.get('message')}")
            
        except Exception as e:
            logger.error(f"Error sending broadcast notification: {str(e)}")
            print(f"❌ Error broadcasting notification: {e}")