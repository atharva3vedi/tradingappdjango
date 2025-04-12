# trading/session_manager.py
# Session manager implementation for Django

from django.core.exceptions import ObjectDoesNotExist
import threading

# Singleton pattern for SessionManager
_session_manager_instance = None
_session_manager_lock = threading.Lock()

def get_session_manager():
    """Get the singleton instance of SessionManager"""
    global _session_manager_instance
    if _session_manager_instance is None:
        with _session_manager_lock:
            if _session_manager_instance is None:
                _session_manager_instance = SessionManager()
    return _session_manager_instance

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def add_session(self, session):
        """Add a session to the manager"""
        self.sessions[session.session_id] = session
    
    def get_session(self, session_id):
        """Get a session by ID"""
        if session_id not in self.sessions:
            raise ObjectDoesNotExist("Session not found")
        return self.sessions[session_id]
    
    def remove_session(self, session_id):
        """Remove a session from the manager"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_all_sessions(self):
        """Get all sessions"""
        return list(self.sessions.values())