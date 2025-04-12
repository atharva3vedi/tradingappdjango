from django.urls import path
from . import views

urlpatterns = [
    # Trading session management
    path('trading/start/', views.start_trading, name='start_trading'),
    path('trading/stop/<str:session_id>/', views.stop_trading, name='stop_trading'),
    
    # Session data retrieval
    path('trading/history/<str:session_id>/', views.get_history, name='get_history'),
    path('trading/status/<str:session_id>/', views.get_session_status, name='get_session_status'),
]