# trading/consumers.py
# WebSocket consumer implementation for Django Channels

import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from .session_manager import get_session_manager

class TradingConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        
        # Add the session ID to the group name
        self.group_name = f'trading_{self.session_id}'
        
        # Join room group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
        
        # Start sending log updates
        asyncio.create_task(self.send_log_updates())
        
    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
    
    async def receive(self, text_data):
        """
        Receive message from WebSocket
        """
        text_data_json = json.loads(text_data)
        message = text_data_json.get('message', '')
        
        if message == 'get_history':
            await self.send_history()
    
    async def send_log_updates(self):
        """Send log updates to the WebSocket client"""
        session_manager = get_session_manager()
        session = session_manager.get_session(self.session_id)
        
        if not session:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Session not found'
            }))
            return
        
        try:
            while True:
                if session.new_logs_available():
                    log = session.get_latest_log()
                    if "type" not in log:
                        continue  # Skip invalid logs
                    
                    # Send structured message based on type
                    if log["type"] == "system":
                        await self.send(text_data=json.dumps({
                            'type': 'system',
                            'data': log["message"]
                        }))
                    elif log["type"] == "step_start":
                        await self.send(text_data=json.dumps({
                            'type': 'progress',
                            'data': {
                                'current': log["step"],
                                'total': len(session.prices)-1
                            }
                        }))
                    elif log["type"] == "action":
                        await self.send(text_data=json.dumps({
                            'type': 'action',
                            'data': log["message"]
                        }))
                    
                await asyncio.sleep(0.05)
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    @database_sync_to_async
    def send_history(self):
        """Send session history to the client"""
        session_manager = get_session_manager()
        session = session_manager.get_session(self.session_id)
        
        if not session:
            return
        
        history_data = {
            'actions': session.get_actions(),
            'portfolio_history': session.get_portfolio_history()
        }
        
        return self.send(text_data=json.dumps({
            'type': 'history',
            'data': history_data
        }))